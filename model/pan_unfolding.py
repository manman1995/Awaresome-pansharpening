import torch
import torch.nn as nn 
import torch.nn.functional as F


import numpy as np
import random

from torch.nn.modules.module import T


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class invPixelShuffle(nn.Module):

    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, y // ratio, x // ratio)

class ExtractFea(torch.nn.Module):
    def __init__(self, channels):
        super(ExtractFea, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)

    def forward(self, frame):
        f0 = F.relu(self.conv1(frame))
        f1 = F.relu(self.conv2(f0))
        f2 = F.relu(self.conv3(f1))
        out = self.conv4(f2)
        # out = self.conv2(f0)
        return out

class blockNL(torch.nn.Module):
    def __init__(self, channels, fs):
        super(blockNL, self).__init__()
        self.channels = channels
        self.fs = fs
        # self.ExtractFea = ExtractFea(channels=self.channels)
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):

        # x_fea = self.ExtractFea(x)

        x_fea = x

        theta = self.t(x_fea).permute(0, 2, 3, 1)#.contiguous()#[b, c, h, w]#[b, h, w,c]
        theta = torch.unsqueeze(theta, dim=-2)  # [b, h, w, 1, c]
        # print(theta.size())

        phi = self.p(x_fea)#[b, c, h, w]
        b, c, h, w = phi.size()
        phi_patches = F.unfold(phi, self.fs, padding=self.fs//2)#[b, c*fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, -1)#[b, c, fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  #[b, c, fs*fs, h, w]
        phi_patches = phi_patches.permute(0, 3, 4, 1, 2)#.contiguous()#[b, h, w, c, fs*fs]
        # print(phi_patches.size())

        att = torch.matmul(theta, phi_patches)# [b, h, w, 1, fs*fs]
        att = self.softmax(att)# [b, h, w, 1, fs*fs]
        # print(att.size())

        g = self.g(x_fea) #[b, 3, h, w]
        g_patches = F.unfold(g, self.fs, padding=self.fs // 2)#[b, 3*fs*fs, hw]
        g_patches = g_patches.view(b, 4, self.fs * self.fs, -1)#[b, 3, fs*fs, hw]
        g_patches = g_patches.view(b, 4, self.fs * self.fs, h, w)#[b, 3, fs*fs, h, w]
        g_patches = g_patches.permute(0, 3, 4, 2, 1)#.contiguous()#[b, h, w, fs*fs, 3]
        # print(g_patches.size())

        out_x = torch.matmul(att, g_patches)  # [1, h, w, 1, 3]
        out_x = torch.squeeze(out_x, dim=-2)# [1, h, w, 3]
        out_x = out_x.permute(0, 3, 1, 2)#.contiguous()
        # print(alignedframe.size())
        return self.w(out_x) + x



class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c, up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        ## x3 00
        ## x2 11
        if up_factor == 2:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=1, output_padding=1),
                nn.Upsample(scale_factor=2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=0, output_padding=0),
                nn.Upsample(scale_factor=3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 4:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Upsample(scale_factor=4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out


class Conv_down(nn.Module):
    def __init__(self, c_in,mid_c, up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 4:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.MaxPool2d(4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 2:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out


class att_spatial(nn.Module):
    def __init__(self, res_num=2):
        super(att_spatial, self).__init__()

        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(res_num):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias=False)

    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale
        
        if self.norm =='batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer, self.act])
        # layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self, base_filter=None, args=None,num_channels=4,mid_channels=64, T = 4):
        super().__init__()

        print("now: pan_unfolding_V4")
        self.up_factor = 4
        G0 = mid_channels
        kSize = 3

        # T = 4
        # todo
        self.conv_u = nn.ModuleList([nn.Sequential(*[                         
            nn.Conv2d(4*(i+1), 64, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(64, 4, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for i in range(T)])

        self.u = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.gama = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])

        # self.gama1 = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        # self.delta1 = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        # self.u1 = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])

        self.conv_up = Conv_up(4, G0, self.up_factor)
        self.conv_down = Conv_down(4, G0, self.up_factor)

        self.rm1 = att_spatial(res_num=3)
        # self.rm2 = att_spatial(res_num=3)

        # # self.NLBlock = nn.ModuleList([blockNL(4, 15) for _ in range(T)])
        self.NLBlock = blockNL(4, 15)

        self.hf_pan = nn.Conv2d(3, 1, 1, padding=0, stride=1)

    def forward(self, lms, b_ms, pan): #l_ms,bms,pan
        # lms B 4 64 64
        # pan B 1 64 64

        hp_pan_2 = pan - F.interpolate(F.interpolate(pan, scale_factor=1/2, mode='bicubic'), scale_factor=2, mode='bicubic') # B 1 256 256

        hp_pan_4 = pan - F.interpolate(F.interpolate(pan, scale_factor=1/4, mode='bicubic'), scale_factor=4, mode='bicubic') # B 1 256 256

        hp_pan_8 = pan - F.interpolate(F.interpolate(pan, scale_factor=1/8, mode='bicubic'), scale_factor=8, mode='bicubic') # B 1 256 256

        pan_hp = self.hf_pan(torch.cat([hp_pan_2, hp_pan_4, hp_pan_8], dim=1))  # B 1 256 256


        hms = torch.nn.functional.interpolate(lms, scale_factor=self.up_factor, mode='bilinear', align_corners=False)   # B 4 256 256
        x = hms

        uk_list = []
        vk_list = []
        outs_list = []

        # decode_u_list = []

        for i in range(len(self.conv_u)):
            if i!=0:
                uk = self.conv_u[i](torch.cat(uk_list + [x], 1))         # B 4 256 256
            else:
                uk = self.conv_u[i](x)         # B 4 256 256
            # uk_list.append(uk)

            # denoising module
            rm1_s2_0 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(uk[:,0,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm1_s2_1 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(uk[:,1,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm1_s2_2 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(uk[:,2,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm1_s2_3 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(uk[:,3,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            decode_u = torch.cat([rm1_s2_0, rm1_s2_1, rm1_s2_2, rm1_s2_3], 1)  # B 4 256 256

            decode_u = decode_u + uk  # B 4 256 256
            uk_list.append(decode_u)


            # NARM
            NL = self.NLBlock(x)        # B 4 256 256
            # vk_list.append(NL)
            if i!=0:
                vk = self.conv_u[i](torch.cat(vk_list+[NL], 1))
            else:
                vk = self.conv_u[i](NL)         # B 4 256 256
            # vk = self.conv_u[i](torch.cat(vk_list, 1))         # B 4 256 256
            # denoising module
            rm2_s2_0 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(vk[:,0,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm2_s2_1 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(vk[:,1,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm2_s2_2 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(vk[:,2,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            rm2_s2_3 = pan_hp + self.rm1(torch.cat([torch.unsqueeze(vk[:,3,:,:],1), pan], 1)) * pan_hp  # B 1 256 256
            decode_v = torch.cat([rm2_s2_0, rm2_s2_1, rm2_s2_2, rm2_s2_3], 1)  # B 4 256 256

            decode_v = decode_v + vk  # B 4 256 256
            vk_list.append(decode_v)

            # iteration
            x = x - self.delta[i]*(self.conv_up(self.conv_down(x)-lms+self.u[i]*(self.conv_down(NL)-lms))+self.eta[i]*(x-decode_u)+self.gama[i]*(NL-decode_v))

            outs_list.append(x)

        return outs_list[-1]# , uk_list[-1], vk_list[-1] # , decoder_list, fea_list
    
    def test(self, device='cpu'):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        input_ms = torch.rand(1, 4, 64, 64)    # 196 为在Embeddings中的 n_patches （14 * 14）
        input_pan = torch.rand(1, 1, 256, 256)
        
        # ideal_out = torch.rand(1, 4, 256, 256)
        
        # out = self.forward(input_ms, None, input_pan)
        
        # assert out.shape == ideal_out.shape
        import torchsummaryX
        torchsummaryX.summary(self, input_ms.to(device), None, input_pan.to(device))

        #
        # from thop import profile
        # flops, params = profile(self, inputs=(input_ms,None, input_pan))
        # print("flops:", flops/1e9, "G")
        # print("params:", params/1e6, "M")

if __name__ == "__main__":


    net3 = pan_unfolding(T=3)
    net3.test()


