from model import common

import torch.nn as nn
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        in_channels = args.n_colors
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool,
                                    resnet.layer1,
                                    resnet.layer2,
                                    resnet.layer3,
                                    resnet.layer4,
                                    resnet.avgpool)
        self.classifier = nn.Linear(512, 1)
        
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
<<<<<<< HEAD
        return self.classifier(x)
=======
        return self.classifier(x)
>>>>>>> df3a6d21ef98a94db9867b542254bd3f827fdb0f
