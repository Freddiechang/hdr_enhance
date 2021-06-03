import torch
import torch.nn as nn
import torch.nn.functional as F

class color_diff_loss(nn.Module):
    def __init__(self, args):
        super(color_diff_loss, self).__init__()
        self.normalization = args.normalization

    def forward(self, x, label):
        if self.normalization != [[0, 0, 0], [1, 1, 1]]:
            std = torch.tensor(self.normalization[1]).view(1,3,1,1)
            mean = torch.tensor(self.normalization[0]).view(1,3,1,1)
            std = std.to(x.device)
            mean = mean.to(x.device)
            x = (x * std) + mean
            label = (label * std) + mean
        c1 = self.color(x)
        c2 = self.color(label)
        l = (c2-c1) ** 2
        return l.mean()

    def color(self, x):
        # rgb to cie xyz
        bitmap = x > 0.04045
        x[bitmap] = ((x[bitmap] + 0.055)/1.055)**2.4
        x[~bitmap] = x[~bitmap]/12.92
        ciex = x[:,0,:,:] * 0.4124 + x[:,1,:,:] * 0.3576 + x[:,2,:,:] * 0.1805
        ciey = x[:,0,:,:] * 0.2126 + x[:,1,:,:] * 0.7152 + x[:,2,:,:] * 0.0722
        ciez = x[:,0,:,:] * 0.0193 + x[:,1,:,:] * 0.1192 + x[:,2,:,:] * 0.9505
        # xyz to lab
        ciex /= 95.047/100
        ciez /= 108.883/100
        x = torch.stack((ciex, ciey, ciez), dim=1)
        bitmap = x > 0.008856
        x[bitmap] = x[bitmap] ** (1/3)
        x[~bitmap] = x[~bitmap] * 7.787 + 16/116
        ciea = 500 * (x[:,0,:,:] - x[:,1,:,:])
        cieb = 500 * (x[:,1,:,:] - x[:,2,:,:])
        return torch.stack((ciea, cieb), dim=1)
