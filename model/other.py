from model import common
from option import args

import torch
import torch.nn as nn
import torch.nn.functional as F

conv = common.default_conv

class Segmentation(nn.Module):
    def __init__(self, args):
        super(Segmentation, self).__init__()
        self.conv = conv(3, args.seg_feats, 3)
        #maybe not a good idea to use pooling
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(args.seg_feats, args.seg_feats)
        self.fc2 = nn.Linear(args.seg_feats, args.seg_feats)
        self.fc3 = nn.Linear(args.seg_feats, args.seg_feats)
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        # N,C,H,W
        x = self.conv(x)
        # N,1,H,W
        #x = self.pool(x)
        x = self.fc1(x.transpose(1, 3))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.transpose(1, 3)
        x = x.argmax(dim = 1)
        return x
        
    

class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        self.conv1 = conv(args.seg_feats, args.seg_feats, 3)
        self.conv2 = conv(args.seg_feats, args.seg_feats, 3)
        self.conv3 = conv(args.seg_feats, 3, 3)
    
    def forward(self, x, seg):
        seg_map = []
        #TODO: append to list then pt.stack
        for i in range(args.seg_feats):
            seg_map.append(seg == i)
        seg_map = torch.stack(seg_map, dim=1)
        x = x * seg_map
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x