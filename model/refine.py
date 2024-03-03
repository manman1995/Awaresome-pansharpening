import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import math
from torch.nn import init
import os
import torchvision.transforms.functional as tf


class DenseModule(nn.Module):
    def __init__(self, channel):
        super(DenseModule, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(channel*4, channel, 1, 1, 0)
        self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        x_final = self.conv4(torch.cat([x,x1,x2,x3],1))

        return x_final

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.process = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
        
        

class Refine(nn.Module):

    def __init__(self,n_feat,out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             # CALayer(n_feat,4),
             # CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class Refine1(nn.Module):

    def __init__(self,in_channels,panchannels,n_feat):
        super(Refine1, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             # CALayer(n_feat,4),
             # CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=in_channels-panchannels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


