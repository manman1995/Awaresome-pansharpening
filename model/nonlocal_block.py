import torch
from torch import nn
from torch.nn import functional as F
from .base_net import *

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias,0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            self.W_pan = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W_pan[0].weight)
            nn.init.constant(self.W_pan[0].bias, 0)
            nn.init.constant(self.W_pan[1].weight, 0)
            nn.init.constant(self.W_pan[1].bias, 0)
       
        else:
            self.W_pan = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W_pan.weight)
            nn.init.constant(self.W_pan.bias, 0)

            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.phi_pan = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        ## PAN
        self.g_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g_pan.weight)
        nn.init.constant(self.g_pan.bias,0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.g_pan = nn.Sequential(self.g_pan, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
                self.phi_pan = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
                self.phi_pan = nn.Sequential(self.phi_pan, max_pool(kernel_size=2))
        
    def forward(self, x, x_pan):
        output = self._embedded_gaussian(x, x_pan)
        return output

    def _embedded_gaussian(self, x, x_pan):
        batch_size = x.size(0)
        patch_size = x.size(2)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, thw/scale^2, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, 0.5c, thw/scale^2)
        # f=>(b, thw/scale^2, 0.5c)dot(b, 0.5c, thw/scale^2) = (b, thw/scale^2, thw/scale^2)
        theta_x = self.theta(x)
        theta_x = self.up(theta_x)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x)
        phi_x = self.up(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_y = self.W(y)
        
        # PAN
        x1 = x_pan
        g_pan_x = self.g_pan(x1).view(batch_size, self.inter_channels, -1)
        g_pan_x = g_pan_x.permute(0, 2, 1)
        
        phi_pan_x = self.phi_pan(x1)
        phi_pan_x = self.up(phi_pan_x)
        phi_pan_x = phi_pan_x.view(batch_size, self.inter_channels, -1)
        
        f_pan = torch.matmul(theta_x, phi_pan_x)
        f_pan_div_C = F.softmax(f_pan, dim=-1)
        y_pan = torch.matmul(f_pan_div_C, g_pan_x)
        y_pan = y_pan.permute(0, 2, 1).contiguous()
        y_pan = y_pan.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_pan_y = self.W_pan(y_pan)
        
        z = torch.cat([W_y, W_pan_y], 1)

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)
