""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.hdr_illu_target = (args.hdr_illu_target - max(args.normalization[0]))
        self.illu_offset = args.hdr_illu_target if args.hdr_illu_target < 0.5 else 1 - args.hdr_illu_target
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
        self.attn1 = SelfAttn(128 // factor, '')
        self.up4 = Up(128, 64, bilinear)
        self.attn2 = SelfAttn(64, '')
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        i1 = x.max(dim=1)[0].unsqueeze(1)
        i1 = (i1 - self.hdr_illu_target).abs() + self.illu_offset
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
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x, _ = self.attn1(x)
        x = self.up4(x, x1)
        x, _ = self.attn2(x)
        x = self.outc(x)
        return x