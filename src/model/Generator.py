import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from https://github.com/milesial/Pytorch-UNet


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    # using LeakyRELU with negative slope 0.1 as in Rivenson et al.

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # modified max pooling to average pooling as in Rivenson et al.

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=3, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)



    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        #x = self.tanh(x)
        #x = (1.0 + self.tanh(x))/2.0
        x = self.sigmoid(x)
        
        return x


# Generator network


class GeneratorUnet(nn.Module):
    def __init__(self, split = False):
        super(GeneratorUnet, self).__init__()
        self.split = split
        factor = 2

        # self.inc = DoubleConv(1, 32)  # 1 or 3
        # Real parameters
#         self.inc1 = DoubleConv(1, 32)
#         self.inc2 = DoubleConv(1, 32)
#         self.inc3 = DoubleConv(1, 32)
#         self.inc4 = DoubleConv(1, 32)
#         self.inc5 = DoubleConv(1, 32)
#         self.inc6 = DoubleConv(1, 32)
#         self.inc7 = DoubleConv(1, 32)
        
#         self.down1 = Down(32*7, 64*7)
#         self.down2 = Down(64*7, 128*7)
#         self.down3 = Down(128*7, 256*7)
#         self.down4 = Down(256*7, 512*7 // factor)
#         self.up1 = Up(512*7, 256*7 // factor)
#         self.up2 = Up(256*7, 128*7 // factor)
#         self.up3 = Up(128*7, 64*7 // factor)
#         self.up4 = Up(64*7, 32*7)
#         self.outc1 = OutConv(32*7, 1)
#         self.outc2 = OutConv(32*7, 1)
#         self.outc3 = OutConv(32*7, 1)
        
        # Tiny parameters
        self.inc1 = DoubleConv(1, 16)
        self.inc2 = DoubleConv(1, 16)
        self.inc3 = DoubleConv(1, 16)
        self.inc4 = DoubleConv(1, 16)
        self.inc5 = DoubleConv(1, 16)
        self.inc6 = DoubleConv(1, 16)
        self.inc7 = DoubleConv(1, 16)
        
        self.down1 = Down(16*7, 32*7)
        self.down2 = Down(32*7, 64*7)
        self.down3 = Down(64*7, 128*7)
        self.down4 = Down(128*7, 256*7 // factor)

        self.up1 = Up(256*7, 128*7 // factor)
        self.up2 = Up(128*7, 64*7 // factor)
        self.up3 = Up(64*7, 32*7 // factor) 
        
        if self.split:
            #self.up3_1 = Up(64*7, 32*7 // factor)
            #self.up3_2 = Up(64*7, 32*7 // factor)
            #self.up3_3 = Up(64*7, 32*7 // factor)
            self.up4_1 = Up(32*7, 16*7)
            self.up4_2 = Up(32*7, 16*7)
            self.up4_3 = Up(32*7, 16*7)
        else:
            self.up4 = Up(32*7, 16*7)
            
        self.outc1 = OutConv(16*7, 1)
        self.outc2 = OutConv(16*7, 1)
        self.outc3 = OutConv(16*7, 1)

    def forward(self, input1, input2, input3, input4, input5, input6, input7):
        # Idea of computing multi-input-multi-output from Ferdian et al (4DflowNET)
        # convolve each input respectively (6 x1s)
#         ins = [input1, input2, input3, input4, input5, input6, input7]
#         first_activations = [inconv(inp) for inp, inconv in zip(ins, self.inc_list)]
#         print('inc :' + str(inc1.size()))
        # now concat 6 inputs
    
        inc1 = self.inc1(input1)
        inc2 = self.inc2(input2)
        inc3 = self.inc3(input3)
        inc4 = self.inc4(input4)
        inc5 = self.inc5(input5)
        inc6 = self.inc6(input6)
        inc7 = self.inc7(input7)
        
        x_concat = torch.cat((inc1, inc2, inc3, inc4, inc5, inc6, inc7), dim=1)
#         print('concat :' + str(x_concat.size()))
        x2 = self.down1(x_concat)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        if self.split:
            #ssemi_out1 = self.up3_1(x, x2)
            #ssemi_out2 = self.up3_2(x, x2)
            #ssemi_out3 = self.up3_3(x, x2)
            
            semi_out1 = self.up4_1(x, x_concat)
            semi_out2 = self.up4_2(x, x_concat)
            semi_out3 = self.up4_3(x, x_concat)
            
            out1 = self.outc1(semi_out1)
            out2 = self.outc2(semi_out2)
            out3 = self.outc3(semi_out3)
        else:
            x = self.up4(x, x_concat)
            # now multi ouput for 3 channels respectively
            out1 = self.outc1(x)
            out2 = self.outc2(x)
            out3 = self.outc3(x)

        return out1, out2, out3
