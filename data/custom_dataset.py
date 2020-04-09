from os import listdir
from os.path import isfile, join, isdir
from torch.multiprocessing import Queue
import threading

from torch.utils.data import Dataset
from torchvision import transforms as tf
from PIL import Image



class MyDataset(Dataset):
    def __init__(self, data_root, normalization=False, mode='train', resize=False, annotation_type='float', totensor=True):
        self.mode = mode
        # transform for image, annotation
        t = [[], []]

        if resize:
            t[0].append(tf.Resize(resize))
            t[1].append(tf.Resize(resize))

        if normalization:
            t[0].append(tf.Normalize(normalization[0], normalization[1]))
            if annotation_type == 'float':
                t[1].append(tf.Normalize(normalization[0], normalization[1]))

        if totensor:
            t[0].append(tf.ToTensor())
            if annotation_type == 'float':
                t[1].append(tf.ToTensor())

        self.transform = [tf.Compose(t[0]), tf.Compose(t[1])]
        
        if mode == 'val':
            tmp_path = join(data_root, 'images', 'val')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

            tmp_path = join(data_root, 'annotations', 'val')
            self.annotations_path = tmp_path
            self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
        elif mode == 'train':
            tmp_path = join(data_root, 'images', 'train')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

            tmp_path = join(data_root, 'annotations', 'train')
            self.annotations_path = tmp_path
            self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
        else:
            tmp_path = join(data_root, 'images', 'test')
            self.images_path = tmp_path
            self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = join(self.images_path, self.images[idx])
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
