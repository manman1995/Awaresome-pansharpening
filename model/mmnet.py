import torch.nn as nn
import torch
from torch.autograd import Variable
from .nonlocal_block import *

import warnings
warnings.filterwarnings('ignore')

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None,groups=1):
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
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, groups=groups,bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,groups=groups,bias=self.bias)

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

class ResnetBlock(nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None,groups=1):
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
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias,groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias,groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad,self.conv2, self.normlayer, self.act])#
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out

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

class C_RB(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1,
    bias=False, activation='prelu', norm=None,n_resblocks=2):
        super(C_RB, self).__init__()
        res_block= [
            ConvBlock(input_channel, output_channel, kernel_size, stride, padding, activation=activation, 
            norm=norm, bias=bias),
        ]
        for i in range(n_resblocks):
            res_block.append(ResnetBlock(output_channel, kernel_size, stride, padding, bias=bias, activation=activation,norm=norm))
        #res_block.append(ConvBlock(output_channel, output_channel, kernel_size, stride, padding, activation=activation,norm=norm, bias=bias))
        self.res_block = nn.Sequential(*res_block)
    def forward(self,x):
        return self.res_block(x)

class RB(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1,
    bias=False, activation='prelu', norm=None,n_resblocks=2):
        super(RB, self).__init__()
        res_block= []
        for i in range(n_resblocks):
            res_block.append(ResnetBlock(output_channel, kernel_size, stride, padding, bias = bias, activation=activation, norm=norm))
        self.res_block = nn.Sequential(*res_block)
    def forward(self,x):
        return self.res_block(x)

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())

class U_block(nn.Module):
    def __init__(self,in_channel=4,mid_channel=4):
        super(U_block,self).__init__()
        self.conv1=ConvBlock(in_channel+4,mid_channel, 5, 1, 2, activation='prelu', norm=None, bias = False)
        self.conv2=RB(mid_channel, in_channel, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.panhead=ConvBlock(1,3, 5, 1, 2, activation='prelu', norm=None, bias = False)
    def forward(self,input,pan):
        pan_f=self.panhead(pan)
        pan_f=torch.cat((pan_f,pan),dim=1)
        x=torch.cat((input,pan_f),dim=1)
        x=self.conv1(x)
        U=self.conv2(x)
        mem_f=torch.cat((U,x,input),dim=1)
        return U,mem_f 
class Hl_block(nn.Module):
    def __init__(self,in_channel=4,mid_channel=4,o_channel=4):
        super(Hl_block,self).__init__()
        self.downconv=ConvBlock(in_channel,mid_channel,5, 4, 2, activation='prelu', norm=None, bias = False)
        self.upconv=nn.ConvTranspose2d(mid_channel,o_channel,5,4,1,1,bias=False) 
        self.act = nn.PReLU(init=0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
    def forward(self,H,U,L,mem):
        if mem is not None:
            H_f=torch.cat((H,mem),dim=1)
        else:
            H_f=H
        DKH=self.downconv(H_f)
        H=H-(self.alpha*self.act(self.upconv(L-DKH))+self.beta*(H-U))
        mem_f=torch.cat((H_f,H),dim=1)
        return H,mem_f 
class V_block(nn.Module):
    def __init__(self,in_channel=4,mid_channel=4):
        super(V_block,self).__init__()
        self.conv1=ConvBlock(in_channel+4,mid_channel,5, 1, 2, activation='prelu', norm=None, bias = False)
        self.conv2 = RB(mid_channel, in_channel, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.panhead = ConvBlock(1, 3, 5, 1, 2, activation='prelu', norm=None, bias=False)
    def forward(self,input,pan):
        pan_f=self.panhead(pan)
        pan_f=torch.cat((pan_f,pan),dim=1) 
        x=torch.cat((input,pan_f),dim=1)
        x=self.conv1(x)
        V=self.conv2(x)
        mem_f=torch.cat((V,x,input),dim=1)
        return V,mem_f 
class Hp_block(nn.Module):
    def __init__(self,in_channel=4,mid_channel=4,o_channel=4):
        super(Hp_block,self).__init__()
        self.downconv=ConvBlock(in_channel,mid_channel,5, 4, 2, activation='prelu', norm=None, bias = False)
        self.upconv=nn.ConvTranspose2d(mid_channel,o_channel,5,4,1,1,bias=False)  #ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize
        self.act = nn.PReLU(init=0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
    def forward(self,SH,V,L,mem):
        if mem is not None:
            H_f=torch.cat((SH,mem),dim=1)
        else:
            H_f=SH
        DKSH=self.downconv(H_f)
        SH=SH-(self.alpha*self.act(self.upconv(L-DKSH))+self.beta*(SH-V))
        mem_f=torch.cat((H_f,SH),dim=1)
        return SH,mem_f

class vanilla_stage(nn.Module):
    def __init__(self):
        super(vanilla_stage,self).__init__()
        self.U=U_block(in_channel=4,mid_channel=4)
        self.Hl=Hl_block(in_channel=4,mid_channel=4)
        self.V=V_block(in_channel=4,mid_channel=4)        
        self.Hp=Hp_block(in_channel=4,mid_channel=4)
        #self.H2SH=ConvBlock(4,4,3, 1, 1, activation='prelu', norm=None, bias = False)
        #self.H2SH = C_RB(4, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.H2SH_Hin=ConvBlock(4,4,3, 2, 1, activation='prelu', norm=None, bias = False)
        self.H2SH_PANin=ConvBlock(1,4,3, 2, 1, activation='prelu', norm=None, bias = False)
        self.H2SH =NONLocalBlock2D(in_channels=4, mode='embedded_gaussian')
        self.H2SH_out=C_RB(8, 4, 1, 1, 0, bias=False, activation='prelu', norm=None, n_resblocks=1)
        self.H2SH_outup=nn.ConvTranspose2d(4,4,3,2,1,1,bias=False)  #ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize

    def forward(self,H,L,pan):
        U,U_mem_out=self.U(H,pan)
        H,Hl_mem_out=self.Hl(H,U,L,None)

        SH=self.H2SH_Hin(H)
        pan_in=self.H2SH_PANin(pan)
        SH=self.H2SH(SH,pan_in)
        SH=self.H2SH_out(SH)
        SH=self.H2SH_outup(SH)

        V,V_mem_out=self.V(SH,pan)
        SH,Hp_mem_out=self.Hp(SH,V,L,None)
        return SH,U_mem_out,Hl_mem_out,V_mem_out,Hp_mem_out

class Net(nn.Module):
    def __init__(self, base_filter=None, args=None,num_channels=4,stage=3):
        super(Net,self).__init__()
        self.stage_num=stage
        self.stage0=vanilla_stage()
        ## memory flow
        self.Uconv1=C_RB(12,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)
        self.cellU=ConvLSTMCell((128,128),8,8,[3,3],False)
        self.Uconv2=C_RB(8,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)

        self.Hlconv1=C_RB(12,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)
        self.cellHl=ConvLSTMCell((128,128),8,8,[3,3],False)
        self.Hlconv2=C_RB(8,4,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)

        self.Vconv1=C_RB(12,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)
        self.cellV=ConvLSTMCell((128,128),8,8,[3,3],False)
        self.Vconv2=C_RB(8,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)

        self.Hpconv1=C_RB(12,8,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)
        self.cellHp=ConvLSTMCell((128,128),8,8,[3,3],False)
        self.Hpconv2=C_RB(8,4,3,1,1,bias=False,activation='prelu',norm=None,n_resblocks=2)

        ## information flow
        self.Uconv3 =C_RB(12, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.H2SH_Hin=ConvBlock(4,4,3, 2, 1, activation='prelu', norm=None, bias = False)
        self.H2SH_PANin=ConvBlock(1,4,3, 2, 1, activation='prelu', norm=None, bias = False)
        self.H2SH =NONLocalBlock2D(in_channels=4, mode='embedded_gaussian')
        self.H2SH_out=C_RB(8, 4, 1, 1, 0, bias=False, activation='prelu', norm=None, n_resblocks=1)
        self.H2SH_outup=nn.ConvTranspose2d(4,4,3,2,1,1,bias=False)  #ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize
        self.Vconv3 =C_RB(12, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        
        self.U_block=U_block(in_channel=4,mid_channel=4)
        self.Hl_block=Hl_block(in_channel=8,mid_channel=4,o_channel=4)
        self.V_block=V_block(in_channel=4,mid_channel=4)
        self.Hp_block=Hp_block(in_channel=8,mid_channel=4,o_channel=4)
    def forward(self,l_ms,bms,pan):
        H=bms

        u_h,u_c=self.cellU.init_hidden(batch_size=pan.size(0))
        Hl_h,Hl_c=self.cellHl.init_hidden(batch_size=pan.size(0))
        v_h,v_c=self.cellV.init_hidden(batch_size=pan.size(0))
        Hp_h,Hp_c=self.cellHp.init_hidden(batch_size=pan.size(0))

        SH,U_mem_f,Hl_mem_f,V_mem_f,Hp_mem_f=self.stage0(H,l_ms,pan)

        for i in range(self.stage_num):
            ###memory###
            if i !=0:
                Hl_mem_f=self.Hlconv1(Hl_mem_f)
                Hp_mem_f=self.Hpconv1(Hp_mem_f)
            U_mem_f=self.Uconv1(U_mem_f)
            V_mem_f=self.Vconv1(V_mem_f)

            u_h,u_c=self.cellU(U_mem_f,cur_state=[u_h, u_c])
            Hl_h,Hl_c=self.cellHl(Hl_mem_f,cur_state=[Hl_h, Hl_c])
            v_h,v_c=self.cellV(V_mem_f,cur_state=[v_h, v_c])
            Hp_h,Hp_c=self.cellHp(Hp_mem_f,cur_state=[Hp_h, Hp_c])

            U_mem_f=self.Uconv2(u_h)  
            Hl_mem_f=self.Hlconv2(Hl_h)
            V_mem_f=self.Vconv2(v_h)
            Hp_mem_f=self.Hpconv2(Hp_h)

            ###main stream###
            H_f=torch.cat((SH,U_mem_f),dim=1)
            H_f=self.Uconv3(H_f)
            U,U_mem_f=self.U_block(H_f,pan)

            H,Hl_mem_f=self.Hl_block(SH,U,l_ms,Hl_mem_f)

            SH = self.H2SH_Hin(H)
            pan_in = self.H2SH_PANin(pan)
            SH = self.H2SH(SH, pan_in)
            SH = self.H2SH_out(SH)
            SH = self.H2SH_outup(SH)+H

            H_f=torch.cat((SH,V_mem_f),dim=1)
            H_f=self.Vconv3(H_f)
            V,V_mem_f=self.V_block(H_f,pan)

            SH,Hp_mem_f=self.Hp_block(SH,V,l_ms,Hp_mem_f)

        return SH


if __name__ == "__main__":

    from torchvision.transforms import Compose, ToTensor
    def transform():
        return Compose([
            ToTensor(),
        ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model=ConvLSTMCell((128,128),8,8,[3,3],True) 
    model2=vanilla_stage()
    model3=Net()
    model3.cuda()
    model2.cuda()
    model.cuda()
    model.eval()
    img=torch.ones((2,4,128,128)).cuda()
    l_ms=torch.ones((2,4,32,32)).cuda()
    pan=torch.ones((2,1,128,128)).cuda()
    t_img=torch.ones((4,8,128,128)).cuda()
    v_h,v_c=model.init_hidden(batch_size=4)
    o_h,o_c=model(t_img,[v_h,v_c])
    SH,U_mem_out,Hl_mem_out,V_mem_out,Hp_mem_out=model2(img,l_ms,pan)
    print("V_mem_out:",V_mem_out.shape)
    model3(l_ms,img,pan)



