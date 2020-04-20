from os import listdir
from os.path import isfile, join, isdir
from torch.multiprocessing import Queue
import threading

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image

from option import args

class Enlighten(Dataset):
    def __init__(self, args, mode='train'):
        data_root = args.dir_data
        normalization = args.normalization
        totensor = args.totensor
        resize = args.resize
        size = [args.img_height, args.img_width]
        annotation_type = args.annotation_type
        self.mode = mode
        
        # transform for image, annotation
        t = [[], []]

        if resize:
            t[0].append(tf.Resize(size))
            t[1].append(tf.Resize(size))

            
        if totensor:
            t[0].append(tf.ToTensor())
            if annotation_type == 'float':
                t[1].append(tf.ToTensor())


        if normalization != [0, 0, 0, 1, 1, 1]:
            t[0].append(tf.Normalize(normalization[0], normalization[1], inplace=True))
            if annotation_type == 'float':
                t[1].append(tf.Normalize(normalization[0], normalization[1], inplace=True))


        self.transform = [tf.Compose(t[0]), tf.Compose(t[1])]
        
        if self.mode == 'val':
            tmp_path = join(data_root, 'trainA')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

            tmp_path = join(data_root, 'trainB')
            self.annotations_path = tmp_path
            self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
        elif self.mode == 'train':
            tmp_path = join(data_root, 'trainA')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

            tmp_path = join(data_root, 'trainB')
            self.annotations_path = tmp_path
            self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
        else:
            tmp_path = join(data_root, 'testA')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            tmp_path = join(data_root, 'testB')
            self.images_path = tmp_path
            self.images += sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

        self.length = [len(self.images), len(self.annotations)]


    def __len__(self):
        return self.length[1]

    def __getitem__(self, idx):
        img_path = join(self.images_path, self.images[idx%self.length[0]])
        anno_path = join(self.annotations_path, self.annotations[idx])


        image = Image.open(img_path)
        image = self.transform[0](image)

        if self.mode != 'test':
            annotation = Image.open(anno_path)
            annotation = self.transform[1](annotation)
        # transform for image, annotation


        sample = {'image': image, 'annotation': annotation}

        return sample


class DataPreFetcher(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


def make_data_loader(args):
    train_dataset = MyDataset(args)
    t = args.data_mode
    args.datamode = 'test'
    test_dataset = MyDataset(args)
    args.datamode = t
    class Loader():
        def __init__(self):
            self.loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=4, drop_last=True)
            self.loader_test  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=4, drop_last=True)
    return Loader()