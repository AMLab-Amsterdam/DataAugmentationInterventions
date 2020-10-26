import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistRotatedDistDa(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True, transform=None, rng_state=0):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        self.transform = transform
        self.rng_state = rng_state

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.y_to_categorical = torch.eye(10)
        self.d_to_categorical = torch.eye(4)

        self.imgs, self.labels = self._get_data()
        len_train = int(0.8*len(self.imgs))

        if self.train:
            self.imgs = self.imgs[:len_train]
            self.labels = self.labels[:len_train]
        else:
            self.imgs = self.imgs[len_train:]
            self.labels = self.labels[len_train:]

    def _get_data(self):
        mnist_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                  train=True,
                                                                  download=self.download,
                                                                  transform=transforms.ToTensor()),
                                                   batch_size=60000,
                                                   shuffle=False)

        for i, (x, y) in enumerate(mnist_loader):
            mnist_imgs = x
            mnist_labels = y

        # Get 10 random ints between 80 and 160
        np.random.set_state(self.rng_state)
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

        if self.transform is not None: # data augmentation
            return self.to_tensor(self.transform(transforms.functional.rotate(x, self.thetas[d]))), self.y_to_categorical[y], self.d_to_categorical[self.d_label]
        else:
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d])), self.y_to_categorical[y], self.d_to_categorical[self.d_label]