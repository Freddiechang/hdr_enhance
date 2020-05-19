import torch
import torch.nn as nn
import torch.nn.functional as F

class HDR_loss(nn.Module):
    def __init__(self, args):
        super(HDR_loss, self).__init__()
        self.normalization = args.normalization

    def forward(self, x, label):
        e = self.exposure(x)
        s = self.saturation(x)
        c = self.contrast(x)
        r1 = e * s * c
        e = self.exposure(label)
        s = self.saturation(label)
        c = self.contrast(label)
        r2 = e * s * c
        l = (r2-r1) ** 2
        return l.sum()

    def exposure(self, x):
        return x.max(dim=1)[0].unsqueeze(1)

    def saturation(self, x):
        cmax = x.max(dim=1)[0]
        cmin = x.min(dim=1)[0]
        s = cmax - cmin
        return s.unsqueeze(1)

    def contrast(self, x):
        sk = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0.]])/3
        k = torch.stack((sk, sk, sk)).unsqueeze(dim=0)
        k = k.to(x.device)
        return F.conv2d(x, k, padding=1)
        