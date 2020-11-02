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
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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

    def forward(self, x):
        return self.conv(x)


# Generator network


class GeneratorUnet(nn.Module):
    def __init__(self):
        super(GeneratorUnet, self).__init__()
        factor = 2

        self.inc = DoubleConv(1, 64)  # 1 or 3
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, input1, input2, input3, input4, input5, input6):
        # Idea of computing multi-input-multi-output from Ferdian et al (4DflowNET)
        # convolve each input respectively (6 x1s)
        inc1 = self.inc(input1)
        inc2 = self.inc(input2)
        inc3 = self.inc(input3)
        inc4 = self.inc(input4)
        inc5 = self.inc(input5)
        inc6 = self.inc(input6)
        # now concat 6 inputs
        x_concat = torch.cat((inc1, inc2, inc3, inc4, inc5, inc6), dim=1)
        x2 = self.down1(x_concat)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x_concat)
        # now multi ouput for 3 channels respectively
        out1 = self.outc(x)
        out2 = self.outc(x)
        out3 = self.outc(x)

        return out1, out2, out3