import torch
import torch.nn as nn

from option import args
from model.edsr import EDSR
from model.other import Fusion
from model.other import Segmentation
from model.unet import UNet

url = 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt'

def make_model(args):
    return Enhance(args)


class Enhance(nn.Module):
    def __init__(self, args):
        super(Enhance, self).__init__()
        self.enhance = UNet(args)
        #self.segmentation = Segmentation(args)
        #self.fusion = Fusion(args)
        self.partial_load = args.partial_load
        self.url = url
    
    def forward(self, x):
        e = self.enhance(x)
        #s = self.segmentation(x)
        #x = self.fusion(e, s)
        return e + x

    def load(self, state_dict, strict=False):
        if self.partial_load:
            self.enhance.load_state_dict(state_dict, strict)
        else:
            self.load_state_dict(state_dict, strict)

