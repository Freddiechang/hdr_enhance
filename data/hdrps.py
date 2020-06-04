from os import listdir
from os.path import isfile, join, isdir
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image
import cv2
import numpy as np
import rawpy

from option import args

class HDRPS(Dataset):
    def __init__(self, args, mode='train'):
        data_root = join(args.dir_data, "HDRPS")
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
        
        tmp_path = join(data_root)
        self.images_path = tmp_path
        self.images = sorted([f for f in listdir(tmp_path) if f != 'HDR'])

        tmp_path = join(data_root, 'HDR')
        self.annotations_path = tmp_path
        self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
        
        self.length = [len(self.images), len(self.annotations)]


    def __len__(self):
        if self.mode != 'unpaired':
            return self.length[0]
        else:
            return self.length[1]

    def __getitem__(self, idx):
        # transform from exr to normal image
        # val * 12.92 if val <= 0.0031308 else 1.055 * val**(1.0/2.4) - 0.055
        if self.mode != 'unpaired':
            images = sorted(listdir(join(self.images_path, self.images[idx])))
            img_path = join(self.images_path, self.images[idx], random.choice(images))
            anno_path = join(self.annotations_path, self.images[idx] + '.exr')


        if img_path[-3:] == 'NEF':
            image = rawpy.imread(img_path)
            image = image.postprocess()
            image = Image.fromarray(image)
        else:
            image = Image.open(img_path)
        image = self.transform[0](image)

        if self.mode in ['test', 'train']:
            annotation = cv2.imread(anno_path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYCOLOR)
            amap = annotation <= 0.0031308
            annotation[amap] = annotation[amap] * 12.92
            annotation[np.logical_not(amap)]= annotation[np.logical_not(amap)] ** (1.0/2.4) * 1.055 - 0.055
            annotation = np.clip(annotation, 0, 1)
            annotation = (annotation * 255).astype(np.uint8)
            annotation = Image.fromarray(annotation)
            annotation = self.transform[1](annotation)
        # transform for image, annotation

        sample = {'image': image, 'annotation': annotation, 'filename': self.images[idx]}

        return sample

