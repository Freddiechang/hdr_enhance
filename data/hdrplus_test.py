from os import listdir
from os.path import isfile, join, isdir
from torch.multiprocessing import Queue
import threading
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image
import rawpy

from option import args

class HDRPLUS_TEST(Dataset):
    def __init__(self, args, mode='train'):
        data_root = join(args.dir_data, "hdrplus")
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


        if normalization != [[0, 0, 0], [1, 1, 1]]:
            t[0].append(tf.Normalize(normalization[0], normalization[1], inplace=True))
            if annotation_type == 'float':
                t[1].append(tf.Normalize(normalization[0], normalization[1], inplace=True))


        self.transform = [tf.Compose(t[0]), tf.Compose(t[1])]
        
        tmp_path = join(data_root, 'bursts')
        self.images_path = tmp_path
        images = sorted([f for f in listdir(tmp_path)])

        tmp_path = join(data_root, 'results_20171023')
        self.annotations_path = tmp_path
        annotations = sorted([f for f in listdir(tmp_path)])
        
        if self.mode == 'test':
            self.images = ['bee7_20150829_155229_854']
            self.annotations = ['bee7_20150829_155229_854']
        elif self.mode == 'train':
            self.images = images[:-300]
            self.annotations = annotations[:-300]
        else:
            self.images = images
            self.annotations = annotations
            

        self.length = len(self.images)


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images = [f for f in listdir(join(self.images_path, self.images[idx])) if 'payload' in f]
        if self.mode == "test":
            img_path = join(self.images_path, self.images[idx], images[len(images)//2])
        else:
            img_path = join(self.images_path, self.images[idx], random.choice(images))
        anno_path = join(self.annotations_path, self.annotations[idx], 'final.jpg')


        image = rawpy.imread(img_path)
        image = image.postprocess()
        image = Image.fromarray(image)
        image = self.transform[0](image)

        if self.mode in ['test', 'train']:
            annotation = Image.open(anno_path)
            annotation = self.transform[1](annotation)
        # transform for image, annotation

        sample = {'image': image, 'annotation': annotation, 'filename': self.images[idx]}

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
