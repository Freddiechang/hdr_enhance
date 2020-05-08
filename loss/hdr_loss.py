import torch
import torch.nn as nn
import torch.nn.functional as F

class HDR_loss(nn.Module):
    def __init__(self, args):
        super(HDR_loss, self).__init__()
        self.normalization = args.normalization

    def forward(self, x):
        pass

    def exposure(self, x):
        pass

    def saturation(self, x):
        pass

    def contrast(self, x):
        pass