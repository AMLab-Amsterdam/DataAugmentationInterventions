import numpy as np

from PIL import Image, ImageFilter

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms


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
    import torchvision
    import PIL

    seed = 0
    da = 'saturation'
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    kwargs = {'num_workers': 8, 'pin_memory': True}

    # Train
    domain_list_train = ['art_painting', 'cartoon', 'photo', 'sketch']

    transform_dict = {'brightness': torchvision.transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
                      'contrast': torchvision.transforms.ColorJitter(brightness=0, contrast=10.0, saturation=0, hue=0),
                      'saturation': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=10.0, hue=0),
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

    transforms_pacs = transforms.Compose([
        transform_dict[da],
        transforms.ToTensor(),
    ])

    train_loader = data_utils.DataLoader(
        PacsDataDataAug('./kfold/', domain_list=domain_list_train, mode='train', transform=transforms_pacs),
        batch_size=128,
        shuffle=True,
        **kwargs)

    y_array = np.zeros(7)
    d_array = np.zeros(4)

    for i, (x, y, d) in enumerate(train_loader):

        # y_array += y.sum(dim=0).cpu().numpy()
        # d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            n = min(x.size(0), 48)
            save_image(x[:n].cpu(),
                       '__pacs_train_' + da + '.png', nrow=8)

            break

    print(y_array, d_array)
    print('\n')
