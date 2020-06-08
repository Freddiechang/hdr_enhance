import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class self_preserve_loss(nn.Module):
    def __init__(self, args):
        super(self_preserve_loss, self).__init__()
        self.normalization = args.normalization
        vgg = models.vgg11(pretrained=True)
        self.vgg = vgg.features[0:17]
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, label):
        if self.args.normalization != [[0, 0, 0], [1, 1, 1]]:
            std = torch.tensor(self.args.normalization[1]).view(1,3,1,1)
            mean = torch.tensor(self.args.normalization[0]).view(1,3,1,1)
            std = std.to(x.device)
            mean = mean.to(x.device)
            x = (x * std) + mean
            label = (label * std) + mean
        x = self.vgg(x)
        label = self.vgg(label)
        diff = label - x
        diff = (diff ** 2).mean()
        return diff
        