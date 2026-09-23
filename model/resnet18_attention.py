import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MultiScaleAttentionBlock(nn.Module):

    def __init__(self, planes, downsample=None):
        super(MultiScaleAttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv_1 = conv3x3(scale_width, scale_width)
        self.bn_1 = norm_layer(scale_width)
        self.conv_2 = conv3x3(scale_width, scale_width)
        self.bn_2 = norm_layer(scale_width)
        self.conv_3 = conv3x3(scale_width, scale_width)
        self.bn_3 = norm_layer(scale_width)
        self.conv_4 = conv3x3(scale_width, scale_width)
        self.bn_4 = norm_layer(scale_width)
        
        self.conv_1_back = conv3x3(scale_width, scale_width)
        self.bn_1_back = norm_layer(scale_width)
        self.conv_2_back = conv3x3(scale_width, scale_width)
        self.bn_2_back = norm_layer(scale_width)
        self.conv_3_back = conv3x3(scale_width, scale_width)
        self.bn_3_back = norm_layer(scale_width)
        self.conv_4_back = conv3x3(scale_width, scale_width)
        self.bn_4_back = norm_layer(scale_width)
        
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sp_x = torch.split(x, self.scale_width, 1)

        out_1 = self.conv_1(sp_x[0])
        out_1 = self.bn_1(out_1)
        out_1_relu = self.relu(out_1)

        out_2 = self.conv_2(out_1_relu + sp_x[1])
        out_2 = self.bn_2(out_2)
        out_2_relu = self.relu(out_2)

        out_3 = self.conv_3(out_2_relu + sp_x[2])
        out_3 = self.bn_3(out_3)
        out_3_relu = self.relu(out_3)

        out_4 = self.conv_4(out_3_relu + sp_x[3])
        out_4 = self.bn_4(out_4)
        
        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)  
        
        out_1_back = self.conv_1_back(sp_x[3])
        out_1_back = self.bn_1_back(out_1_back)
        out_1_relu_back = self.relu(out_1_back)

        out_2_back = self.conv_2_back(out_1_relu_back + sp_x[2])
        out_2_back = self.bn_2_back(out_2_back)
        out_2_relu_back = self.relu(out_2_back)

        out_3_back = self.conv_3_back(out_2_relu_back + sp_x[1])
        out_3_back = self.bn_3_back(out_3_back)
        out_3_relu_back = self.relu(out_3_back)

        out_4_back = self.conv_4_back(out_3_relu_back + sp_x[0])
        out_4_back = self.bn_4_back(out_4_back)

        out_back = torch.cat([out_1_back, out_2_back, out_3_back, out_4_back], dim=1)
        
        out = out + out_back
        
        attention_score = self.sigmoid(out)

        out = attention_score * x
        out += x
        out = self.relu(out)

        return out
        

class ThreeDAttentionBlock(nn.Module):

    def __init__(self, planes, downsample=None):
        super(ThreeDAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = x * attention
        out += x
        out = self.relu(out)

        return out
        

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool )
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.ChannelGate(x)
        out = self.SpatialGate(out)
        out += x
        out = self.relu(out)

        return out
        
        
class Decomposer(nn.Module):
    def __init__(self, nfeat):
        super(Decomposer, self).__init__()
        self.nfeat = nfeat
        '''
        self.embed_layer1 = nn.Sequential(nn.Conv2d(nfeat*2, nfeat*2, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(nfeat*2),
                                         nn.ReLU(),
                                         )
        self.attention_layer = ThreeDAttentionBlock(nfeat*2)
        self.embed_layer2 = nn.Sequential(nn.Conv2d(nfeat*2, nfeat*2, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(nfeat*2),
                                         nn.ReLU(),
                                         )
        '''
        
    def forward(self, x):
        # embedded = self.embed_layer1(x)
        # embedded = self.attention_layer(embedded)
        # embedded = self.embed_layer2(embedded)
        # rele, irre = torch.split(embedded, [int(self.nfeat), int(self.nfeat)], dim=1)
        rele, irre = torch.split(x, [int(self.nfeat), int(self.nfeat)], dim=1)

        return rele, irre


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
 

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        
        self.decomposer = Decomposer(256)
        
        self.pool_DR = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_DR = nn.Dropout(0.2)
        self.fc_DR = nn.Linear(512, 4)
        
        self.pool_DME = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_DME = nn.Dropout(0.2)
        self.fc_DME = nn.Linear(512, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_DR_feature, x_DME_feature = self.decomposer(x)
        
        x_DR_feature = self.pool_DR(x_DR_feature)
        x_DR_feature = torch.flatten(x_DR_feature, 1)
        output_DR = self.dropout_DR(x_DR_feature)
        output_DR = self.fc_DR(output_DR)
        
        x_DME_feature = self.pool_DME(x_DME_feature)
        x_DME_feature = torch.flatten(x_DME_feature, 1)        
        output_DME = self.dropout_DME(x_DME_feature)
        output_DME = self.fc_DME(output_DME)

        return x_DR_feature, x_DME_feature, output_DR, output_DME


def Resnet_Dis():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model
