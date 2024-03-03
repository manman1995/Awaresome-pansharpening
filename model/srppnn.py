#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-11 20:37:09
LastEditTime: 2020-12-09 23:12:50
Description: Super-Resolution-Guided Progressive Pansharpening Based on a Deep Convolutional Neural Network
batch_size = 64, MSE, Adam, 0.0001, patch_size = 64, 2000 epoch, decay 1000, x0.1
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        out_channels = 4
        n_resblocks = 11

        res_block_s1 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        res_block_s2 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s2.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s2 = nn.Sequential(*res_block_s2)
        
        res_block_s3 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s3.append(Upsampler(2, 32, activation='prelu'))
        res_block_s3.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s3 = nn.Sequential(*res_block_s3)

        res_block_s4 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s4.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s4 = nn.Sequential(*res_block_s4)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):

        hp_pan_4 =  x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1/4, mode='bicubic'), scale_factor=4, mode='bicubic')
        lr_pan = F.interpolate(x_pan, scale_factor=1/2, mode='bicubic')
        hp_pan_2 =  lr_pan - F.interpolate(F.interpolate(lr_pan, scale_factor=1/2, mode='bicubic'), scale_factor=2, mode='bicubic')
        
        s1 = self.res_block_s1(l_ms)
        s1 = s1 + F.interpolate(l_ms, scale_factor=2, mode='bicubic')
        s2 = self.res_block_s2(torch.cat([s1, lr_pan], 1)) + \
            F.interpolate(l_ms, scale_factor=2, mode='bicubic') + hp_pan_2
        s3 = self.res_block_s3(s2) + b_ms
        s4 = self.res_block_s4(torch.cat([s3, x_pan], 1))+ \
            b_ms + hp_pan_4
        
        return s4