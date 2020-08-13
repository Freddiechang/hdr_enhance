import torch
import torch.nn as nn

from option import args
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
        self.normalization = args.normalization
    
    def forward(self, x):
        e = self.enhance(x)
        #s = self.segmentation(x)
        #x = self.fusion(e, s)
        e += x
        e = e - e.min(dim=2)[0].min(dim=2)[0].unsqueeze(2).unsqueeze(3)
        e = e / e.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(3)
        e = e - torch.tensor(self.normalization[0]).view(1, -1, 1, 1).to(e.device)
        return e

    def load(self, state_dict, strict=False):
        if self.partial_load:
            self.enhance.load_state_dict(state_dict, strict)
        else:
            self.load_state_dict(state_dict, strict)

