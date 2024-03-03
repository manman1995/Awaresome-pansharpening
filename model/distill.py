# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine1
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=False):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class InvBlockMscale(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlockMscale, self).__init__()
        self.ops1 = InvBlock(subnet_constructor, channel_num, channel_split_num)
        # self.ops2 = InvBlock(subnet_constructor, channel_num, channel_split_num)
        # self.ops3 = InvBlock(subnet_constructor, channel_num, channel_split_num)
        self.fuse = nn.Conv2d(3*channel_num,channel_num,1,1,0)

    def forward(self,x,rev=False):
        x1 = x
        x2 = F.interpolate(x1,scale_factor=0.5,mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops1(x1)
        x2 = self.ops1(x2)
        x3 = self.ops1(x3)
        x2 = F.interpolate(x2,size=(x1.size()[2],x1.size()[3]),mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x = self.fuse(torch.cat([x1,x2,x3],1))

        return x

class FeatureExtract(nn.Module):
    def __init__(self, channel_in=3, channel_split_num=3, n_feat = 32, subnet_constructor=subnet('DBNet'), block_num=3):
        super(FeatureExtract, self).__init__()
        operations = []

        # current_channel = channel_in
        channel_num = channel_in

        for j in range(block_num):
            b = InvBlockMscale(subnet_constructor, channel_num, channel_split_num)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.fuse = nn.Conv2d((block_num-1)*channel_in,channel_in,1,1,0)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x, rev=False):
        out = x  # x: [N,3,H,W]
        outfuse = out
        for i,op in enumerate(self.operations):
            out = op.forward(out, rev)
            if i == 1:
                outfuse = out
            elif i > 1:
                outfuse = torch.cat([outfuse,out],1)
            if i < 2:
                out = out+x
        outfuse = self.fuse(outfuse)

        return outfuse


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class PNNPro(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer=6):
        super(PNNPro, self).__init__()
        self.conv_pan = nn.Conv2d(pan_channels, n_feat//2, 1, 1, 0)
        self.conv_ms = nn.Conv2d(ms_channels, n_feat // 2, 1, 1, 0)
        self.extract = FeatureExtract(n_feat, n_feat//2)
        self.refine = Refine1(ms_channels + pan_channels, pan_channels, n_feat)

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        if ms.shape[2] == pan.shape[2]:
            mHR = ms
        else:
            mHR = upsample(ms, M, N)
        panf = self.conv_pan(pan)
        mHRf = self.conv_ms(mHR)
        finput = torch.cat([panf, mHRf], dim=1)
        fmid = self.extract(finput)
        HR = self.refine(fmid)+mHR

        return HR, finput, fmid


################################################################################ Unfolding Network

class Netn(nn.Module):
    def __init__(self, ms_channels,pan_channels,n_feat,n_layer):
        super(Netn, self).__init__()
        self.process = PNNPro(ms_channels,pan_channels,n_feat,n_layer)
        keys = torch.load('/home/jieh/Projects/PAN_Sharp/GPPNN/training/logs/GPPNN5/2/best_net.pth')['net']
        self.process.load_state_dict(keys)
        for p in self.parameters():
            p.requires_grad = False
        self.unfold1 = LRMS(ms_channels, pan_channels,n_feat)
        self.unfold2 = LRMS(ms_channels, pan_channels,n_feat)

    def forward(self, ms, pan=None):
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)

        ms_en = self.process(ms, pan)
        fold1 = self.unfold1(ms_en,mHR,pan)
        fold2 = self.unfold2(fold1,mHR,pan)

        return fold2

class Net1(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net1, self).__init__()

        ms_channels = 4
        pan_channels = 1
        n_layer = 8
        n_feat = 8

        self.process = GPPNNPro(ms_channels,pan_channels,n_feat,n_layer)
        self.unfold1 = LRMS(ms_channels, pan_channels,n_feat)
        self.unfold2 = LRMS(ms_channels, pan_channels,n_feat)
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

    def forward(self, ms, b_ms, pan=None):
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)

        ms_en = self.process(ms, pan)
        fold1 = self.unfold1(ms_en,ms,pan)+ms_en
        fold2 = self.unfold2(fold1,ms,pan)+fold1

        return fold2+mHR


###################################################################################    Unfolding Block

class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size // 2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
        )

    def forward(self, input):
        return self.basic_unit(input)


class LRMS(nn.Module):
    def __init__(self, ms_channels, pan_channels,n_feat, kernel_size=1):
        super(LRMS, self).__init__()

        self.get_LR = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_HR_residual = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels)

        self.get_PAN = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)
        self.get_PAN_residual = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size)
        # self.prox = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)

    def forward(self, HR, LR, PAN):
        _, _, M, N = HR.shape
        # print(HR.shape)
        _, _, m, n = LR.shape
        # print(LR.shape)

        PAN_hat = self.get_PAN(HR)
        PAN_Residual = PAN - PAN_hat
        HR_Residual_1 = self.get_PAN_residual(PAN_Residual)

        LR_hat = upsample(self.get_LR(HR), m, n)
        LR_Residual = LR - LR_hat
        HR_Residual_2 = upsample(self.get_HR_residual(LR_Residual), M, N)

        HR = self.prox(HR + HR_Residual_1 + HR_Residual_2)
        return HR

class Net2(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net2, self).__init__()

        ms_channels = 4
        pan_channels = 1
        n_layer = 8
        n_feat = 8

        self.process = GPPNNPro(ms_channels, pan_channels, n_feat, n_layer)
        #self.unfold1 = LRMS(ms_channels, pan_channels,n_feat)
        #self.unfold2 = LRMS(ms_channels, pan_channels,n_feat)
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

    def forward(self, ms, b_ms, pan=None):
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        #mHR = upsample(ms, M, N)

        ms_en, finput, fmid = self.process(ms, pan)
        #fold1 = self.unfold1(ms_en,ms,pan)+ms_en
        #fold2 = self.unfold2(fold1,ms,pan)+fold1

        return ms_en, finput, fmid

class Net(nn.Module):
    def __init__(self,
                 num_channels, base_filter, args):
        super(Net, self).__init__()

        ms_channels = 4
        pan_channels = 1
        n_layer = 8
        n_feat = 8
        self.conv_pan = nn.Conv2d(pan_channels, n_feat//2, 1, 1, 0)
        self.conv_ms = nn.Conv2d(ms_channels, n_feat // 2, 1, 1, 0)
        self.extract = FeatureExtract(n_feat, n_feat//2)
        self.refine = Refine1(ms_channels + pan_channels, pan_channels, n_feat)

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

    def forward(self, ms, b_ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        if ms.shape[2] == pan.shape[2]:
            mHR = ms
        else:
            mHR = upsample(ms, M, N)
        panf = self.conv_pan(pan)
        mHRf = self.conv_ms(mHR)
        finput = torch.cat([panf, mHRf], dim=1)
        fmid = self.extract(finput)
        HR = self.refine(fmid)+mHR

        return HR, finput, fmid