""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        n_classes = args.seg_feats
        bilinear = True

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        i1 = x.max(dim=1).unsqueeze(1)
        i1 = (i1 - 0.1).abs()
        x1 = self.inc(x) * i1
        i2 = F.interpolate(i1, scale_factor=0.5)
        x2 = self.down1(x1) * i2
        i3 = F.interpolate(i2, scale_factor=0.5)
        x3 = self.down2(x2) * i3
        i4 = F.interpolate(i3, scale_factor=0.5)
        x4 = self.down3(x3) * i4
        i5 = F.interpolate(i4, scale_factor=0.5)
        x5 = self.down4(x4) * i5
        x = self.up1(x5, x4) * i4
        x = self.up2(x, x3) * i3
        x = self.up3(x, x2) * i2
        x = self.up4(x, x1) * i1
        logits = self.outc(x) * i1
        return logits
