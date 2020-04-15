import torch
import torch.nn as nn

from option import args
from model.edsr import EDSR
from model.other import Fusion
from model.other import Segmentation


def make_model(args):
    return Enhance(args)


class Enhance(nn.Module):
    def __init__(self, args):
        super(Enhance, self).__init__()
        self.enhance = EDSR(args)
        self.segmentation = Segmentation(args)
        self.fusion = Fusion(args)
    
    def forward(self, x):
        e = self.enhance(x)
        s = self.segmentation(x)
        x = self.fusion(e, s)
        return x


def make_model(args):
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = Enhance(args)
    return model.to(device)