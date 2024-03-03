import torch.nn as nn
import torch

BatchNorm2d = nn.BatchNorm2d

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

class CConv(nn.Module):
    def __init__(self, in_channels, out_channels,bias=True,g_inchannels=3):
        super(CConv, self).__init__()
        self.split=in_channels
        inter_channels=in_channels
        setattr(self, 'fc{}', nn.Sequential(*[
            nn.Conv2d(in_channels=g_inchannels, out_channels=inter_channels, 
                        kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(init=0.5),
            nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels*2, 
                        kernel_size=3, stride=1, padding=1, bias=bias),
        ]))
        setattr(self, 'conv{}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                    kernel_size=3, stride=1, padding=1, bias=bias))

        self.gate=nn.PReLU(init=0.5)
    def forward(self,input,g_input):
        scale, translation = torch.split(getattr(self, 'fc{}')(g_input), (self.split,self.split), dim=1)
        output=getattr(self, 'conv{}')(self.gate(input*torch.sigmoid(scale) + translation))
        return output

class Bandmodulation(nn.Module):
    def __init__(self,in_channels, out_channels,f_channels=4,bias=True,split=4,copy_wight=3,g_inchannels=1):
        super(Bandmodulation,self).__init__()
        self.shortcut=True
        if in_channels !=out_channels:
             self.shortcut=False
        self.num_split=split
        self.copy_wight=copy_wight
        self.split_list=[] 
        self.fusion_list=[]
        in_split,f_split=in_channels//split,f_channels//split
        out_split=out_channels//split
        for i in range(self.num_split):
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split+f_split, out_channels=in_split, 
                                                        kernel_size=3, stride=1, padding=1, bias=bias))
            setattr(self, 'cconv{}'.format(i),CConv(in_channels=in_split,out_channels=out_split,g_inchannels=copy_wight*g_inchannels))
            self.split_list.append(in_split)
            self.fusion_list.append(f_split)
    def forward(self,input,f_input,g_input):
        t_input = torch.split(input, self.split_list, dim=1) #tuple (4,[1,1,128,128])
        t_fusion= torch.split(f_input, self.fusion_list, dim=1)
        output = [] 
        g_x=g_input
        for i in range(self.copy_wight-1):
            g_x=torch.cat((g_x,g_input),1)

        for i in range(self.num_split):
            split_fusion=getattr(self,'conv{}'.format(i))(torch.cat((t_input[i],t_fusion[i]),1)) 
            output.append(getattr(self,'cconv{}'.format(i))(split_fusion,g_x))
        if not self.shortcut:
            return torch.cat(output,1)
        return torch.cat(output, 1)+input


class FMM_MSB(nn.Module):
    def __init__(self, inchannels,f_channels=4,copy_wight=1,g_inchannels=1):
        super(FMM_MSB, self).__init__()
        self.BMM=Bandmodulation(inchannels,inchannels,f_channels=f_channels,copy_wight=copy_wight,g_inchannels=g_inchannels)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(inchannels*5, inchannels, 1, padding=0),
            nn.ReLU(inchannels)
        )

        n_resblocks = 0
        n_resblocks1 = 1
        res_block_s1 = [
            ConvBlock(inchannels, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=1), 
        ]
        for i in range(n_resblocks1):
            res_block_s1.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None,groups=1))
        self.res_block_s1 = nn.Sequential(*res_block_s1)
        res_block_s2 = [
            ConvBlock(inchannels, 24, 1, 1, 0, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.res_block_s2 = nn.Sequential(*res_block_s2)

    def forward(self,x,f_x,g_x):
        BMM_f =self.BMM(x,f_x,g_x)
        BMM_f =self.res_block_s1(BMM_f)
        conv1 = self.conv_1(BMM_f)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(BMM_f)
        conv4 = self.conv_4(conv3)
        F_DCM = self.conv_5(torch.cat([BMM_f, conv1, conv2, conv3, conv4], dim=1))
        F_DCM = self.res_block_s2(F_DCM)
        F_out = F_DCM + x
        return F_out

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args): 
        super(Net, self).__init__()
        inchannel=num_channels*2
        interplanes=inchannel*3 
        self.head = ConvBlock(inchannel, interplanes, 9, 1, 4, activation='prelu', norm=None, bias = False)
        self.head2=ConvBlock(1, 3, 9, 1, 4, activation='prelu', norm=None, bias = False)
        self.conv1=ConvBlock(24,24 , 5, 1, 2, activation='prelu', norm=None, bias = False)
        self.FMM_MSB_1=FMM_MSB(24)
        self.FMM_MSB_2=FMM_MSB(24)
        self.gconv1=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=4)
        self.gconv2=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=4)

        self.conv2=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False)
        self.output_conv = ConvBlock(24, num_channels, 5, 1, 2, activation=None, norm=None, bias = False)
    def forward(self,l_ms,bms,pan):
        pan0=self.head2(pan)
        pan0=torch.cat((pan, pan0), 1) 
        cbms=torch.cat((bms, pan0), 1) 
        x0=self.head(cbms) 
        x1=self.conv1(x0)  
        x=self.FMM_MSB_1(x1,bms,pan)
        x=self.FMM_MSB_2(x,bms,pan) 
        x=torch.add(x,x1)
        x=self.gconv1(x)
        x=self.conv2(x) 
        x=torch.add(x,x0)
        x=self.gconv2(x)
        x=self.output_conv(x)+bms       
        return x


if __name__ == "__main__":

    from torchvision.transforms import Compose, ToTensor
    def transform():
        return Compose([
            ToTensor(),
        ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model=Net(4)
    model.eval()
    img=torch.ones((1,4,128,128))
    l_ms=torch.ones((1,4,32,32))
    pan=torch.ones((1,1,128,128))
    output_end=model(l_ms,img,pan)
    print(output_end.shape)