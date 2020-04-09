from option import args
from model.edsr import EDSR
from loss.discriminator import Discriminator
from loss import adversarial_loss

import torch as pt
#print(args)
a=EDSR(args)
b=pt.randn((10,3,64,64))
print(a(b).shape)