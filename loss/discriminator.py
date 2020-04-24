from model import common

import torch.nn as nn

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        in_channels = args.n_colors
        out_channels = 64
        depth = 2
        temp = [self.base_block(in_channels, out_channels)]
        for i in range(depth - 1):
            temp.append(self.base_block(out_channels, out_channels))
        self.features = nn.Sequential(*temp)
        class_in_feats = out_channels * args.img_height * args.img_width // 2**(depth*2)
        temp = [
            nn.Linear(class_in_feats, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*temp)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
    
    def base_block(self, in_channels, out_channels, stride=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return block