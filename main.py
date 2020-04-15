import utility
from option import args
import model
import loss
from data.enlighten import make_data_loader
from trainer import Trainer

def main():
    checkpoint = utility.checkpoint(args)
    # TODO: modify to allow loading checkpoint
    # if using GAN, both model and loss need to be loaded
    _data_loader = make_data_loader(args)
    _model = model.Model(args, checkpoint)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    _trainer = Trainer(args, _data_loader, _model, _loss, checkpoint)
    while not _trainer.terminate():
        _trainer.train()
        _trainer.test()

    checkpoint.done()

if __name__ == '__main__':
    main()