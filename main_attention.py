import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
import random
import torchsummary
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from dataset import *
from collections import OrderedDict
from model.resnet18_attention import Resnet_Dis
from model.module import CLUBSample

my_whole_seed = 111
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


checkpoint_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('-classes_DR', default=5, type=int, metavar='N', help='bumber of DR grading classes')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=40, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=checkpoint_path, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--gpu', type=str, default='0,1')
args = parser.parse_args()


def main(time_str):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    checkpoint_path = './checkpoint/' + time_str + 'model.pth'
    best_checkpoint_path = './checkpoint/' + time_str + 'model_best.pth'

    # create model and load pretrained parameters
    model = Resnet_Dis()
    
    # checkpoint = torch.load("/home/zouwei/code/checkpoint/resnet50-19c8e357.pth")
    checkpoint = torch.load("/home/zouwei/code/checkpoint/resnet18-5c106cde.pth")
    pretrained = checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model = torch.nn.DataParallel(model).cuda()

    model.module.fc_DR = torch.nn.Linear(256, args.classes_DR).cuda()
    model.module.fc_DME = torch.nn.Linear(256, 3).cuda()
    
    print(torchsummary.summary(model, (3, 224, 224)))

    # estimator = CLUBSample(x_dim=256, y_dim=256, hidden_size=256).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion_mean = nn.LogSoftmax(dim=1).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_estimate = torch.optim.Adam(estimator.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler_estimate = torch.optim.lr_scheduler.StepLR(optimizer_estimate, step_size=30, gamma=0.5)
    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    train_dataset = Dataset(txt_root='/home/zouwei/dataset/IDRiD/Disease_Grading/dataset_train.txt',
                            transform=torchvision.transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                               ])
                            )
    test_dataset = Dataset(txt_root='/home/zouwei/dataset/IDRiD/Disease_Grading/dataset_test.txt',
                           transform=torchvision.transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                           ])
                           )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc_DR, train_los_DR, train_acc_DME, train_los_DME = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc_DR, val_los_DR, val_acc_DME, val_los_DME, joint_acc = validate(val_loader, model, criterion, epoch, args)

        scheduler.step()
        # scheduler_estimate.step()

        recorder.update(epoch, train_los_DR, train_acc_DR, train_los_DME, train_acc_DME, val_los_DR, val_acc_DR, val_los_DME, val_acc_DME)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = joint_acc > best_acc
        best_acc = max(joint_acc, best_acc)

        print('Current best accuracy: ', best_acc)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, checkpoint_path, best_checkpoint_path)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('An Epoch Time:' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
                         
    # switch to train model            
    losses_DR = AverageMeter('Loss_DR', ':.4f')
    losses_DME = AverageMeter('Loss_DME', ':.4f')
    # losses_mi = AverageMeter('Loss_mi', ':.4f')
    top1_DR = AverageMeter('Accuracy_DR', ':6.3f')
    top1_DME = AverageMeter('Accuracy_DME', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses_DR, losses_DME, top1_DR, top1_DME],
                             prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    # estimator.eval()

    for i, (images, target_DR, target_DME) in enumerate(train_loader):

        images = images.cuda()
        target_DR = target_DR.cuda()
        target_DME = target_DME.cuda()

        # compute output and loss
        x_DR_feature, x_DME_feature, output_DR, output_DME = model(images)

        loss_DR = criterion(output_DR, target_DR)
        loss_DME = criterion(output_DME, target_DME)
        
        # loss_mi = estimator(x_DR_feature, x_DME_feature)

        # measure accuracy and record loss
        acc1_DR, _ = accuracy(output_DR, target_DR, topk=(1, 2))
        acc1_DME, _ = accuracy(output_DME, target_DME, topk=(1, 2))

        losses_DR.update(loss_DR.item(), images.size(0))
        losses_DME.update(loss_DME.item(), images.size(0))
        # losses_mi.update(loss_mi.item(), images.size(0))
        
        top1_DR.update(acc1_DR[0], images.size(0))
        top1_DME.update(acc1_DME[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss = loss_DR + loss_DME + 0.001 * loss_mi
        loss = loss_DR + loss_DME
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
    '''
    # switch to train estimator
    for times in range(1, 4):
        losses_CLUB = AverageMeter('Loss_CLUB', ':.4f')
        progress_estimate = ProgressMeter(len(train_loader),
                                          [losses_CLUB],
                                          prefix="Epoch_MI: [{}] Time:[{}]".format(epoch, times))
        model.eval()
        estimator.train()

        for i, (images, target_DR, target_DME) in enumerate(train_loader):
            images = images.cuda()

            x_DR_feature, x_DME_feature, _, _ = model(images)

            loss_club = estimator.learning_loss(x_DR_feature, x_DME_feature)

            losses_CLUB.update(loss_club.item(), images.size(0))

            optimizer_estimate.zero_grad()
            loss = loss_club
            loss.backward()
            optimizer_estimate.step()

            if i % args.print_freq == 0:
                progress_estimate.display(i)
    '''
                                              
    return top1_DR.avg, losses_DR.avg, top1_DME.avg, losses_DME.avg


def validate(val_loader, model, criterion, epoch, args):
    losses_DR = AverageMeter('Loss_DR', ':.4f')
    losses_DME = AverageMeter('Loss_DME', ':.4f')
    top1_DR = AverageMeter('Accuracy_DR', ':6.3f')
    top1_DME = AverageMeter('Accuracy_DME', ':6.3f')
    top1 = AverageMeter('Accuracy_joint', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses_DR, losses_DME, top1_DR, top1_DME, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        target_all_DR = torch.zeros(1)
        target_all_DME = torch.zeros(1)
        output_all_DR = torch.zeros(1)
        output_all_DME = torch.zeros(1)

        for i, (images, target_DR, target_DME) in enumerate(val_loader):
            images = images.cuda()
            target_DR = target_DR.cuda()
            target_DME = target_DME.cuda()
            
            if i==0:
                target_all_DR = target_DR
                target_all_DME = target_DME
            else:
                target_all_DR = torch.cat([target_all_DR, target_DR], dim=0)
                target_all_DME = torch.cat([target_all_DME, target_DME], dim=0)

            # compute output and loss
            _, _, output_DR, output_DME = model(images)
            loss_DR = criterion(output_DR, target_DR)
            loss_DME = criterion(output_DME, target_DME)
            
            if i==0:
                output_all_DR = output_DR
                output_all_DME = output_DME
            else:
                output_all_DR = torch.cat([output_all_DR, output_DR], dim=0)
                output_all_DME = torch.cat([output_all_DME, output_DME], dim=0)

            # measure accuracy and record loss
            acc1_DR, _ = accuracy(output_DR, target_DR, topk=(1, 2))
            acc1_DME, _ = accuracy(output_DME, target_DME, topk=(1, 2))
            acc1 = joint_accuracy(output_DR, target_DR, output_DME, target_DME)

            losses_DR.update(loss_DR.item(), images.size(0))
            losses_DME.update(loss_DME.item(), images.size(0))

            top1_DR.update(acc1_DR[0], images.size(0))
            top1_DME.update(acc1_DME[0], images.size(0))
            top1.update(acc1.item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        auc_DR = multi_class_auc(target_all_DR.cpu().data.numpy(), torch.softmax(output_all_DR, dim=1).cpu().data.numpy(), args.classes_DR) * 100.0
        auc_DME = multi_class_auc(target_all_DME.cpu().data.numpy(), torch.softmax(output_all_DME, dim=1).cpu().data.numpy(), 3) * 100.0
        f1_DR = f1_score(target_all_DR.cpu().data.numpy(), torch.argmax(output_all_DR, axis=1).cpu().data.numpy(), average="macro") * 100.0
        f1_DME = f1_score(target_all_DME.cpu().data.numpy(), torch.argmax(output_all_DME, axis=1).cpu().data.numpy(), average="macro") * 100.0
        print('***')
        print('Accuracy_DR:{top1_DR.avg:.3f} Accuracy_DME:{top1_DME.avg:.3f} Accuracy_joint:{top1.avg:.3f}'.format(top1_DR=top1_DR, top1_DME=top1_DME, top1=top1))
        print('AUC_DR:{auc_DR:.3f} F1_DR:{f1_DR:.3f} AUC_DME:{auc_DME:.3f} F1_DME:{f1_DME:.3f}'.format(auc_DR=auc_DR, auc_DME=auc_DME, f1_DR=f1_DR, f1_DME=f1_DME))
        print('***')
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write('Accuracy_DR:{top1_DR.avg:.3f} Accuracy_DME:{top1_DME.avg:.3f} Accuracy_joint:{top1.avg:.3f}'.format(top1_DR=top1_DR, top1_DME=top1_DME, top1=top1) + '\n' +
            'AUC_DR:{auc_DR:.3f} F1_DR:{f1_DR:.3f} AUC_DME:{auc_DME:.3f} F1_DME:{f1_DME:.3f}'.format(auc_DR=auc_DR, auc_DME=auc_DME, f1_DR=f1_DR, f1_DME=f1_DME) + '\n')
    return top1_DR.avg, losses_DR.avg, top1_DME.avg, losses_DME.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '  '.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def joint_accuracy(output_1, target_1, output_2, target_2):

    batch_size = target_1.size(0)
    _, pred_1 = output_1.topk(1, 1, True, True)
    pred_1 = pred_1.t()
    correct_1 = pred_1.eq(target_1.view(1, -1).expand_as(pred_1))

    _, pred_2 = output_2.topk(1, 1, True, True)
    pred_2 = pred_2.t()
    correct_2 = pred_2.eq(target_2.view(1, -1).expand_as(pred_2))

    correct = correct_1.float() * correct_2.float()
    correct = torch.sum(correct)
    res = correct.mul_(100.0 / batch_size)
    return res
    

def multi_class_auc(all_target, all_output, num_c = None):

    all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    auc_sum = []

    for num_class in range(0, num_c):
        auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
        auc_sum.append(auc)

    auc = sum(auc_sum) / float(len(auc_sum))

    return auc
    
        
class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 4), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 4), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss_DR, train_acc_DR, train_loss_DME, train_acc_DME, val_loss_DR, val_acc_DR, val_loss_DME, val_acc_DME):
        self.epoch_losses[idx, 0] = train_loss_DR * 20
        self.epoch_losses[idx, 1] = train_loss_DME * 20
        self.epoch_losses[idx, 2] = val_loss_DR * 20
        self.epoch_losses[idx, 3] = val_loss_DME * 20
        self.epoch_accuracy[idx, 0] = train_acc_DR
        self.epoch_accuracy[idx, 1] = train_acc_DME
        self.epoch_accuracy[idx, 2] = val_acc_DR
        self.epoch_accuracy[idx, 3] = val_acc_DME
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy-DR', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='train-accuracy-DME', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 2]
        plt.plot(x_axis, y_axis, color='r', linestyle='-', label='valid-accuracy-DR', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 3]
        plt.plot(x_axis, y_axis, color='b', linestyle='-', label='valid-accuracy-DME', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x20-DR', lw=4)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='train-loss-x20-DME', lw=4)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 2]
        plt.plot(x_axis, y_axis, color='r', linestyle=':', label='valid-loss-x20-DR', lw=4)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 3]
        plt.plot(x_axis, y_axis, color='b', linestyle=':', label='valid-loss-x20-DME', lw=4)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    for i in range(2):
        now = datetime.datetime.now()
        time_str = now.strftime("[%m-%d]-[%H-%M]-")
        main(time_str)
