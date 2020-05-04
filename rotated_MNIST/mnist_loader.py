"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistRotated(data_utils.Dataset):
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

        pil_list = []
        for x in mnist_imgs:
            pil_list.append(self.to_pil(x))

        return pil_list, mnist_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        d = np.random.choice(range(len(self.thetas)))

        if self.transform: # data augmentation random rotation by +- 90 degrees
            random_rotation = np.random.choice([-30.0, 0.0, 30.0])
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d] + random_rotation)), self.y_to_categorical[y], \
                   self.d_to_categorical[self.d_label]
        else:
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d])), self.y_to_categorical[y], self.d_to_categorical[self.d_label]


if __name__ == "__main__":
    from torchvision.utils import save_image

    seed = 0

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # mnist_90 = MnistRotated('../dataset/', train=False, thetas=[90.0], d_label=0)
    # train_loader = data_utils.DataLoader(mnist_90,
    #                                      batch_size=100,
    #                                      shuffle=False)
    #
    # y_array = np.zeros(10)
    # d_array = np.zeros(3)
    #
    # for i, (x, y, d) in enumerate(train_loader):
    #     y_array += y.sum(dim=0).cpu().numpy()
    #     d_array += d.sum(dim=0).cpu().numpy()
    #
    #     if i == 0:
    #         n = min(x.size(0), 36)
    #         comparison = x[:n].view(-1, 1, 28, 28)
    #         save_image(comparison.cpu(),
    #                    'rotated_mnist.png', nrow=6)
    #
    # print(y_array, d_array)

    mnist_0 = MnistRotated('../dataset/', train=False, thetas=[0.0], d_label=0)
    train_loader = data_utils.DataLoader(mnist_0,
                                         batch_size=1000,
                                         shuffle=False)

    for i, (x, y, d) in enumerate(train_loader):

        y = y.argmax(-1)

        index = y ==5
        x = x[index]

        if i == 0:
            comparison = x[82].view(-1, 1, 28, 28)
            save_image(comparison.cpu(),
                       'rotated_mnist_2.png', nrow=1)