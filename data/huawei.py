from os import listdir
from os.path import isfile, join, isdir

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image

from option import args

class Huawei(Dataset):
    def __init__(self, args, mode='test'):
        data_root = join(args.dir_data, "huawei")
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
        
        tmp_path = data_root
        self.images_path = tmp_path
        self.images = sorted([f for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
<<<<<<< HEAD

=======
            
>>>>>>> df3a6d21ef98a94db9867b542254bd3f827fdb0f

        self.length = len(self.images)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = join(self.images_path, self.images[idx])


        image = Image.open(img_path)
        if image.mode == "RGBA":
            image.load()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        annotation = image
        image = self.transform[0](image)
<<<<<<< HEAD
        #print(image.shape)
=======
>>>>>>> df3a6d21ef98a94db9867b542254bd3f827fdb0f
        annotation = self.transform[1](annotation)


        sample = {'image': image, 'annotation': annotation, 'filename': self.images[idx]}

        return sample

<<<<<<< HEAD

=======
>>>>>>> df3a6d21ef98a94db9867b542254bd3f827fdb0f
