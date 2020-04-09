import torch
import torch.nn as nn

from option import args
from model import edsr.EDSR
from model import other.Fusion
from model import other.Segmentation


class Enhance(nn.Module):
    def __init__(self, args):
        self.enhance = EDSR(args)
        self.segmentation = Segmentation(args)
        self.fusion = Fusion(args)
    
    def forward(self, x):
        e = self.enhance(x)
        s = self.segmentation(x)
        x = self.fusion(e, s)
        return x
        