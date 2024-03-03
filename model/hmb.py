#!/usr/bin/env python
# coding=utf-8
'''
Author: zm, wjm
Date: 2020-11-25 00:34:50
LastEditTime: 2021-05-31 14:58:29
Description: file content
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from model.nonlocal_block import *

class att_spatial(nn.Module):
    def __init__(self):
        super(att_spatial, self).__init__()
        kernel_size = 7
        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(6):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias = False)
        
    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = F.sigmoid(x_out) # broadcasting
        return scale

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        out_channels = 4
        n_resblocks = 11
        
        # pixel domain
        res_block_s1 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, 1, 3, 1, 1, activation='prelu', norm=None, bias = False))
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
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.res_block_s4 = nn.Sequential(*res_block_s4)
        
        res_block_s4 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s4.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s4 = nn.Sequential(*res_block_s4)      

        self.rm1 = att_spatial()
        self.rm2 = att_spatial()
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
        
    def forward(self, l_ms, b_ms, x_pan):

        hp_pan_4 =  x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1/4, mode='bicubic'), scale_factor=4, mode='bicubic')
        lr_pan = F.interpolate(x_pan, scale_factor=1/2, mode='bicubic')
        hp_pan_2 =  lr_pan - F.interpolate(F.interpolate(lr_pan, scale_factor=1/2, mode='bicubic'), scale_factor=2, mode='bicubic')
      
        # NN, pixel domain
        s1 = self.res_block_s1(l_ms)
        s1 = s1 + F.interpolate(l_ms, scale_factor=2, mode='bicubic')
        s2 = self.res_block_s2(torch.cat([s1, lr_pan], 1))
            
        # residual modification
        rm_s2_0 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:,0,:,:],1), lr_pan], 1)) * hp_pan_2 
        rm_s2_1 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:,1,:,:],1), lr_pan], 1)) * hp_pan_2
        rm_s2_2 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:,2,:,:],1), lr_pan], 1)) * hp_pan_2
        rm_s2_3 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:,3,:,:],1), lr_pan], 1)) * hp_pan_2
        rm_s2_pan = torch.cat([rm_s2_0, rm_s2_1, rm_s2_2, rm_s2_3], 1)
        
        s2 = s2 + F.interpolate(l_ms, scale_factor=2, mode='bicubic') + hp_pan_2
        
        s3 = self.res_block_s3(s2) + b_ms

        s4 = self.res_block_s4(torch.cat([s3, x_pan], 1)) 

        # residual modification
        rm_s4_0 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:,0,:,:],1), x_pan], 1)) * hp_pan_4
        rm_s4_1 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:,1,:,:],1), x_pan], 1)) * hp_pan_4
        rm_s4_2 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:,2,:,:],1), x_pan], 1)) * hp_pan_4
        rm_s4_3 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:,3,:,:],1), x_pan], 1)) * hp_pan_4
        rm_s4_pan = torch.cat([rm_s4_0, rm_s4_1, rm_s4_2, rm_s4_3], 1)
        
        s4 = s4 + b_ms + hp_pan_4 

        return s4
        