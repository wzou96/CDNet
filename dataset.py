import os
import torch
import torchvision
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, txt_root, transform=None):
        super(Dataset, self).__init__
        f = open(txt_root, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            line_split = line.split(' ')
            imgs.append((line_split[0], int(line_split[1]), int(line_split[2])))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        root, label_DR, label_DME = self.imgs[index]
        img = Image.open(root).convert('RGB')    

        if self.transform is not None:
            img = self.transform(img)       

        return img, label_DR, label_DME

    def __len__(self):
        return len(self.imgs)

        