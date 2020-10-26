"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torchvision
import PIL


class MnistAllDaDist(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        transform_dict = {
            'brightness': torchvision.transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
            'contrast': torchvision.transforms.ColorJitter(brightness=0, contrast=1.0, saturation=0, hue=0),
            'saturation': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=1.0, hue=0),
            'hue': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
            'rotation': torchvision.transforms.RandomAffine([0, 359], translate=None, scale=None, shear=None,
                                                            resample=PIL.Image.BILINEAR, fillcolor=0),
            'translate': torchvision.transforms.RandomAffine(0, translate=[0.2, 0.2], scale=None, shear=None,
                                                             resample=PIL.Image.BILINEAR, fillcolor=0),
            'scale': torchvision.transforms.RandomAffine(0, translate=None, scale=[0.8, 1.2], shear=None,
                                                         resample=PIL.Image.BILINEAR, fillcolor=0),
            'shear': torchvision.transforms.RandomAffine(0, translate=None, scale=None,
                                                         shear=[-10., 10., -10., 10.],
                                                         resample=PIL.Image.BILINEAR, fillcolor=0),
            'vflip': torchvision.transforms.RandomVerticalFlip(p=0.5),
            'hflip': torchvision.transforms.RandomHorizontalFlip(p=0.5),
            'none': None,
            }

        self.transforms = torchvision.transforms.Compose([transform_dict['brightness'],
                                                     transform_dict['contrast'],
                                                     transform_dict['saturation'],
                                                     transform_dict['hue'],
                                                     transform_dict['rotation'],
                                                     transform_dict['translate'],
                                                     transform_dict['scale'],
                                                     transform_dict['shear'],
                                                     transform_dict['vflip'],
                                                     transform_dict['hflip']])

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.y_to_categorical = torch.eye(10)
        self.d_to_categorical = torch.eye(4)

        self.imgs, self.labels = self._get_data()


    def _get_data(self):
        mnist_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                  train=self.train,
                                                                  download=self.download,
                                                                  transform=transforms.ToTensor()),
                                                   batch_size=60000,
                                                   shuffle=False)

        for i, (x, y) in enumerate(mnist_loader):
            mnist_imgs = x
            mnist_labels = y

        # Get 10 random ints between 80 and 160
        label_dist = np.random.randint(80, 160, 10)

        mnist_imgs_dist, mnist_labels_dist = [], []
        for i in range(10):
            idx = np.where(mnist_labels == i)[0]
            np.random.shuffle(idx)
            idx = idx[:label_dist[i]] # select the right amount of labels for each class
            mnist_imgs_dist.append(mnist_imgs[idx])
            mnist_labels_dist.append(mnist_labels[idx])

        mnist_imgs_dist = torch.cat(mnist_imgs_dist)
        mnist_labels_dist = torch.cat(mnist_labels_dist)

        pil_list = []
        for x in mnist_imgs_dist:
            pil_list.append(self.to_pil(x))

        return pil_list, mnist_labels_dist

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        d = np.random.choice(range(len(self.thetas)))

        return self.to_tensor(self.transforms(transforms.functional.rotate(x, self.thetas[d]))), self.y_to_categorical[y], self.d_to_categorical[self.d_label]
