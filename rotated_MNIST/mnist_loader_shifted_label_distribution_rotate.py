"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistRotatedDist(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True, transform=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        self.transform = transform

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.y_to_categorical = torch.eye(10)
        self.d_to_categorical = torch.eye(3)

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

        if self.transform: # data augmentation, random rotation by 90 degrees
            random_rotation = np.random.randint(0, 360, 1)
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d] + random_rotation)), self.y_to_categorical[y], self.d_to_categorical[self.d_label]
        else:
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d])), self.y_to_categorical[y], self.d_to_categorical[self.d_label]


if __name__ == "__main__":
    from torchvision.utils import save_image

    seed = 0

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    mnist_0 = MnistRotatedDist('../dataset/', train=True, thetas=[0], d_label=0, transform=True)
    mnist_30 = MnistRotatedDist('../dataset/', train=True, thetas=[30.0], d_label=1, transform=True)
    mnist_60 = MnistRotatedDist('../dataset/', train=True, thetas=[60.0], d_label=2, transform=True)
    mnist = data_utils.ConcatDataset([mnist_0, mnist_30, mnist_60])
    train_loader = data_utils.DataLoader(mnist,
                                         batch_size=100,
                                         shuffle=True)

    y_array = np.zeros(10)
    d_array = np.zeros(3)

    for i, (x, y, d) in enumerate(train_loader):
        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            n = min(x.size(0), 36)
            comparison = x[:n].view(-1, 1, 28, 28)
            save_image(comparison.cpu(),
                       'rotated_mnist_dist.png', nrow=6)

    print(y_array, d_array)