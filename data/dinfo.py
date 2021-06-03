from os import listdir
from os.path import isfile, join, isdir

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image

from option import args

class DINFO(Dataset):
    def __init__(self, args, mode='train'):
        data_root = join(args.dir_data, "DINFO")
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
        
        tmp_path = join(data_root, 'sdr')
        self.images_path = tmp_path
        images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

        tmp_path = join(data_root, 'hdr')
        self.annotations_path = tmp_path
        annotations = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])

        images_ = map(lambda x: ''.join(x.split("SDR")), images)
        anno_ = map(lambda x: ''.join(x.split("HDR")), annotations)
        
        images_ = sorted(list(set(images_).intersection(set(anno_))))
        
        if self.mode == 'test':
            images_ = [f for f in images_ if f[0:3] in ['I01', 'I03', 'A03', 'A04']]
            self.images = sorted(list(map(lambda x: x.replace('__', '_SDR_'), images_)))
            self.annotations = sorted(list(map(lambda x: x.replace('__', '_HDR_'), images_)))
            #self.images = sorted([f for f in listdir(tmp_path) if isfile(join(self.images_path, f))])

            #self.annotations = sorted([f for f in listdir(tmp_path) if isfile(join(self.annotations_path, f))])
        elif self.mode == 'train':
            images_ = [f for f in images_ if f[0:3] not in ['I01', 'I03', 'A03', 'A04']]
            self.images = sorted(list(map(lambda x: x.replace('__', '_SDR_'), images_)))
            self.annotations = sorted(list(map(lambda x: x.replace('__', '_HDR_'), images_)))
        else:
            self.images = sorted(list(map(lambda x: x.replace('__', '_SDR_'), images_)))
            self.annotations = sorted(list(map(lambda x: x.replace('__', '_HDR_'), images_)))
            

        self.length = [len(self.images), len(self.annotations)]


    def __len__(self):
        return self.length[1]

    def __getitem__(self, idx):
        img_path = join(self.images_path, self.images[idx])
        anno_path = join(self.annotations_path, self.annotations[idx])


        image = Image.open(img_path)
        image = self.transform[0](image)

        if self.mode in ['test', 'train']:
            annotation = Image.open(anno_path)
            annotation = self.transform[1](annotation)
        # transform for image, annotation

        sample = {'image': image, 'annotation': annotation, 'filename': self.images[idx]}

        return sample

