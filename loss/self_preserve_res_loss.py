import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class self_preserve_res_loss(nn.Module):
    def __init__(self, args):
        super(self_preserve_res_loss, self).__init__()
        self.normalization = args.normalization
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool,
                                    resnet.layer1,
                                    resnet.layer2,
                                    resnet.layer3)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, x, label):
        if self.normalization != [[0, 0, 0], [1, 1, 1]]:
            std = torch.tensor(self.normalization[1]).view(1,3,1,1)
            mean = torch.tensor(self.normalization[0]).view(1,3,1,1)
            std = std.to(x.device)
            mean = mean.to(x.device)
            x = (x * std) + mean
            label = (label * std) + mean
        x = self.resnet(x)
        label = self.resnet(label)
        diff = label - x
        diff = (diff ** 2).mean()
        return diff
<<<<<<< HEAD
        
=======
        
>>>>>>> df3a6d21ef98a94db9867b542254bd3f827fdb0f
