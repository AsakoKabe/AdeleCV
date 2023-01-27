import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class _Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Inconv, self).__init__()
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(_Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class _Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetTorch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetTorch, self).__init__()
        self.inc = _Inconv(n_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        self.down4 = _Down(512, 512)
        self.up1 = _Up(1024, 256)
        self.up2 = _Up(512, 128)
        self.up3 = _Up(256, 64)
        self.up4 = _Up(128, 64)
        self.outc = _Outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
