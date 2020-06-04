from os import listdir
from os.path import isfile, join, isdir
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image
import rawpy

from option import args

class HDRPLUS(Dataset):
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
            self.images = images[-15:]
            self.annotations = annotations[-15:]
        elif self.mode == 'train':
            self.images = images[:-15]
            self.annotations = annotations[:-15]
        else:
            self.images = images
            self.annotations = annotations
            

        self.length = len(self.images)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images = [f for f in listdir(join(self.images_path, self.images[idx])) if 'payload' in f]
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

