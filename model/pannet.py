#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:47:04
LastEditTime: 2020-12-09 23:12:31
Description: PanNet: A deep network architecture for pan-sharpening (VDSR-based)
2000 epoch, decay 1000 x0.1, batch_size = 128, learning_rate = 1e-2, patch_size = 33, MSE
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

        base_filter = 64
        num_channels = 7
        out_channels = 4
        self.args = args
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias = False)

        self.body = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation=None, norm=None, bias = False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    #torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    #torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):

        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)
        x_f = torch.add(x_f,b_ms)
     
        return x_f
        
