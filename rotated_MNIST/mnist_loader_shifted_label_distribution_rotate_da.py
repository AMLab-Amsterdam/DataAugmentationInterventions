"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from paper_experiments.rotated_MNIST.mnist_loader_da import MnistRotatedDistDa


if __name__ == "__main__":
    from torchvision.utils import save_image
    import torchvision
    import PIL

    seed = 0
    da = 'hflip'

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    transform_dict = {'brightness': torchvision.transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
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

    rng_state = np.random.get_state()
    mnist_0_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[0], d_label=0, transform=transform_dict[da], rng_state=rng_state)
    mnist_0_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[0], d_label=0, transform=None,
                                     rng_state=rng_state)
    rng_state = np.random.get_state()
    mnist_30_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[30.0], d_label=1, transform=transform_dict[da], rng_state=rng_state)
    mnist_30_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[30.0], d_label=1, transform=None,
                                      rng_state=rng_state)
    rng_state = np.random.get_state()
    mnist_60_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[60.0], d_label=2, transform=transform_dict[da], rng_state=rng_state)
    mnist_60_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[60.0], d_label=2,
                                      transform=transform_dict[da], rng_state=rng_state)
    mnist_train = data_utils.ConcatDataset([mnist_0_train, mnist_30_train, mnist_60_train])
    train_loader = data_utils.DataLoader(mnist_train,
                                         batch_size=100,
                                         shuffle=True)

    mnist_val = data_utils.ConcatDataset([mnist_0_val, mnist_30_val, mnist_60_val])
    val_loader = data_utils.DataLoader(mnist_val,
                                         batch_size=100,
                                         shuffle=True)

    y_array = np.zeros(10)
    d_array = np.zeros(3)

    for i, (x, y, d) in enumerate(train_loader):
        # y_array += y.sum(dim=0).cpu().numpy()
        # d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            n = min(x.size(0), 36)
            comparison = x[:n].view(-1, 1, 28, 28)
            save_image(comparison.cpu(),
                       'rotated_mnist_' + da + '.png', nrow=6)

    # print(y_array, d_array)