from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                m = import_module('data.' + d.lower())
                datasets.append(getattr(m, d)(args))

            self.loader_train = dataloader.DataLoader(
                ConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
                drop_last=True
            )

        testsets = []
        for d in args.data_test:
            m = import_module('data.' + d.lower())
            testsets.append(getattr(m, d)(args, mode='test'))

        self.loader_test = dataloader.DataLoader(
                ConcatDataset(testsets),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

