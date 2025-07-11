#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:48:27
LastEditTime: 2020-12-09 23:13:08
Description: PNN (SRCMM-based), Pansharpening by Convolutional Neural Networks
batch_size = 128, learning_rate = 1e-2, patch_size = 33, MSE, 2000 epoch, decay 1000, x0.1, MSE
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
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias = True)

        self.body = ConvBlock(48, 32, 5, 1, 2, activation='relu', norm=None, bias = True)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='relu', norm=None, bias = True)

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

        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)

        return x_f

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_in1 = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_in2 = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_in3 = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1))#,BatchNorm2d(plane)

    def forward(self, x):
        # b, c, h, w = x.size()
        node_in1 = self.node_in1(x)
        node_in2 = self.node_in2(x)
        node_in3 = self.node_in3(x)
        b,c,h,w = node_in1.size()
        node_in1 = node_in1.view(b, c, -1).permute(0, 2, 1)
        node_in3 = node_in3.view(b, c, -1)
        node_in2 = node_in2.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_in3,node_in2)
        AV = self.softmax(AV)
        AV = torch.bmm(node_in1, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out
