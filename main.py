import utility
from option import args
from model import enhance.Enhance
from loss import adversarial_loss.Adversarial
from trainer import Trainer

def main():
    checkpoint = utility.checkpoint(args)
    # TODO: modify to allow loading checkpoint
    # if using GAN, both model and loss need to be loaded
    _data_loader = []
    _model = Enhance(args)
    _loss = Adversarial(args)
    _trainer = Trainer(args, _data_loader, _model, _loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

if __name__ == '__main__':
    main()