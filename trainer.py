import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        # for unpaired training
        self.loss.gan_set_train()
        #######################
        timer_data, timer_model = utility.timer(), utility.timer()

        # TODO: change according to return of dataloader 
        for batch, data in enumerate(self.loader_train):
            lr, hr = self.prepare(data['image'], data['annotation'])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    # TODO: need to rewrite evaluation metrics
    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.model.eval()
        # for unpaired training
        #self.loss.gan_set_eval()
        #######################
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        ########### scale is no longer needed, set to 1
        scale = 1
        idx_scale = 1
        ###########
        for idx_data, d in enumerate(self.loader_test):
            lr, hr = self.prepare(d['image'], d['annotation'])
            sr = self.model(lr)

            
            loss = self.loss(sr, hr)
            #used for unpaired training
            self.ckp.log[-1] += loss
            if self.args.save_gt:
                save_list.extend([lr, hr])

            if self.args.save_results:
                save_list = [lr, hr, sr]
                self.ckp.save_results(d['filename'], save_list)

        #self.ckp.log[-1] /= (idx_data + 1)
        best = self.ckp.log.min(0)
        self.ckp.write_log(
            '[{} x{}]\tLoss: {:.3f} (Best: {:.3f} @epoch {})'.format(
                'test_run',
                scale,
                self.ckp.log[-1],
                best[0],
                best[1] + 1
            )
        )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def test_only(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.model.eval()
        # for unpaired training
        #self.loss.gan_set_eval()
        #######################
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        ########### scale is no longer needed, set to 1
        scale = 1
        ###########
        for idx_data, d in enumerate(self.loader_test):
            lr, hr = self.prepare(d['image'], d['annotation'])
            sr = self.model(lr)
            filename = d['filename'][0]

            if self.args.save_results:
                save_list = [lr, hr, sr]
                self.ckp.save_results(filename, save_list)
        
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:' + str(self.args.select_gpu))
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test_only()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

