import numpy as np

from PIL import Image, ImageFilter

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


# def cropping(image, crop_size, dim1, dim2):
#     """crop the image and pad it to in_size
#     Args :
#         images : numpy array of images
#         crop_size(int) : size of cropped image
#         dim1(int) : vertical location of crop
#         dim2(int) : horizontal location of crop
#     Return :
#         cropped_img: numpy array of cropped image
#     """
#     cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
#     return cropped_img
#
#
# def add_elastic_transform(image, alpha, sigma, pad_size=30, seed=None):
#     """
#     Args:
#         image : numpy array of image
#         alpha : α is a scaling factor
#         sigma :  σ is an elasticity coefficient
#         random_state = random integer
#         Return :
#         image : elastically transformed numpy array of image
#     """
#     image = image.numpy()
#     image_size = int(image.shape[0])
#     image = np.pad(image, pad_size, mode="symmetric")
#     if seed is None:
#         seed = np.randint(1, 100)
#         random_state = np.random.RandomState(seed)
#     else:
#         random_state = np.random.RandomState(seed)
#     shape = image.shape
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
#                          sigma, mode="constant", cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
#                          sigma, mode="constant", cval=0) * alpha
#
#     x, y = np.meshgrid(np.arange(shape[2]), np.arange(shape[1]))
#     indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
#     return cropping(map_coordinates(image, indices, order=1).reshape(shape), image_size, pad_size, pad_size), seed
#
#
# class ElasticDeformation(object):
#     def __init__(self, alpha, sigma, seed):
#         self.alpha = alpha
#         self.sigma = sigma
#         self.seed = seed
#
#     def __call__(self, img):
#         return add_elastic_transform(img, alpha=self.alpha, sigma=self.sigma, seed=self.seed)


class RandomGaussianBlur(object):
    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(np.random.normal(0.0, 0.5, 1)))
        return img


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = np.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img


class PacsDataDataAug(data_utils.Dataset):
    def __init__(self, path, domain_list=None, mode='train', transform=None):
        self.path = path
        self.domain_list = domain_list
        self.mode = mode
        self.transform = transform

        self.train_data, self.train_labels, self.train_domain = self.get_data()

    def get_imgs_and_labels(self, file_name, domain_path):

        with open(file_name, 'r') as data:
            img_paths = []
            labels = []
            for line in data:
                p = line.split()
                img_paths.append(domain_path + p[0])
                labels.append(p[1])

        img_list = []
        label_list = []

        for i, img_path in enumerate(img_paths):
            with open(img_path, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')

            img_list.append(img)
            label_list.append((np.float(labels[i]) - 1.0))

        return img_list, torch.Tensor(np.array(label_list))

    def get_data(self):
        imgs_per_domain_list = []
        labels_per_domain_list = []
        domain_per_domain_list = []

        for i, domain in enumerate(self.domain_list):

            domain_path = self.path + domain

            if self.mode == 'train':
                file_name = domain_path + '_train_kfold.txt'
            elif self.mode == 'val':
                file_name = domain_path + '_crossval_kfold.txt'
            elif self.mode == 'test':
                file_name = domain_path + '_test_kfold.txt'
            else:
                print('unkown mode found')

            imgs, labels = self.get_imgs_and_labels(file_name, self.path)
            domain_labels = torch.zeros(labels.size()) + i

            # append to final list
            imgs_per_domain_list.append(imgs)
            labels_per_domain_list.append(labels)
            domain_per_domain_list.append(domain_labels)

        # One last cat
        # train_imgs = torch.cat(imgs_per_domain_list).squeeze()
        train_imgs = [item for sublist in imgs_per_domain_list for item in sublist]

        train_labels = torch.cat(labels_per_domain_list).long()
        train_domains = torch.cat(domain_per_domain_list).long()

        # Convert to onehot
        y = torch.eye(7)
        train_labels = y[train_labels]

        d = torch.eye(4)
        train_domains = d[train_domains]

        return train_imgs, train_labels, train_domains

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domain[index]

        return self.transform(x), y, d


if __name__ == "__main__":
    from torchvision.utils import save_image

    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    kwargs = {'num_workers': 8, 'pin_memory': True}

    # Train
    domain_list_train = ['art_painting', 'cartoon', 'photo', 'sketch']

    transforms_pacs = transforms.Compose([
        RandomGaussianBlur(),
        transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=1, hue=0.5),
        transforms.RandomGrayscale(p=0.1),
        RandomRotate(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(0, translate=None, scale=(1.0, 1.1), shear=None, resample=False,
                                            fillcolor=0),
        transforms.ToTensor()])

    train_loader = data_utils.DataLoader(
        PacsDataDataAug('./kfold/', domain_list=domain_list_train, mode='train', transform=transforms_pacs),
        batch_size=128,
        shuffle=True,
        **kwargs)

    y_array = np.zeros(7)
    d_array = np.zeros(4)

    for i, (x, y, d) in enumerate(train_loader):

        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            n = min(x.size(0), 48)
            save_image(x[:n].cpu(),
                       'pacs_train_heavy_data_aug.png', nrow=8)

    print(y_array, d_array)
    print('\n')
