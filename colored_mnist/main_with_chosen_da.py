# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
# from paper_experiments.colored_mnist.choose_da_with_domain_classifier import CMNISTWithTransform, MLP, make_environment
import torchvision
import PIL

# Build environments
def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()


def make_environment(images, labels, e):
    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images_zeros = torch.zeros_like(images)
    images = torch.stack([images, images, images_zeros], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.),
        'labels': labels[:, None]}


# Dataset with transform
class CMNISTWithTransform(torch.utils.data.DataLoader):
    def __init__(self, transform, train=True):
        self.transform = transform
        self.to_pil = torchvision.transforms.ToPILImage(mode='RGB')
        self.to_tensor = torchvision.transforms.ToTensor()

        # Load MNIST, make train/val splits, and shuffle train set examples
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        if train:
            mnist = (mnist.data[:50000], mnist.targets[:50000])
        else:
            mnist = (mnist.data[50000:], mnist.targets[50000:])

        rng_state = np.random.get_state()
        np.random.shuffle(mnist[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist[1].numpy())

        env1 = make_environment(mnist[0][::2], mnist[1][::2], 0.2)
        env2 = make_environment(mnist[0][1::2], mnist[1][1::2], 0.1)

        images = torch.cat([env1['images'], env2['images']])
        self.labels = torch.cat([env1['labels'], env2['labels']])
        self.domains = torch.cat([torch.zeros_like(env1['labels']), torch.zeros_like(env2['labels'])+1.0])

        self.pil_images = []
        for image in images:
            self.pil_images.append(self.to_pil(image))

    def __getitem__(self, index):
        image, label, domain = self.pil_images[index], self.labels[index], self.domains[index]
        if self.transform is not None:
            return self.to_tensor(self.transform(image)), label, domain
        else:
            return self.to_tensor(image), label, domain

    def __len__(self):
        return len(self.labels)


# Define and instantiate the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        lin1 = nn.Linear(3 * 14 * 14, args.hidden_dim)
        lin2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        lin3 = nn.Linear(args.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input.view(input.shape[0], 3 * 14 * 14)
        out = self._main(out)
        return F.sigmoid(out)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCELoss()(output, target)
        loss.backward()
        optimizer.step()

        # save_image(data.cpu(),
        #            'baseline.png', nrow=1)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += nn.BCELoss()(output, target).item() # sum up batch loss
            pred = output >= 0.5
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
# parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# Put into data loader
device = torch.device("cuda")
kwargs = {'num_workers': 8, 'pin_memory': False}

# Data aug
transform_dict = {'brightness': torchvision.transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
                  'contrast': torchvision.transforms.ColorJitter(brightness=0, contrast=1.0, saturation=0, hue=0),
                  'saturation': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=1.0, hue=0),
                  'hue': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
                  'rotation': torchvision.transforms.RandomAffine([0, 359], translate=None, scale=None, shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'translate': torchvision.transforms.RandomAffine(0, translate=[0.2, 0.2], scale=None, shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'scale': torchvision.transforms.RandomAffine(0, translate=None, scale=[0.8, 1.2], shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'shear': torchvision.transforms.RandomAffine(0, translate=None, scale=None, shear=[-10., 10., -10., 10.], resample=PIL.Image.BILINEAR, fillcolor=0),
                  'vflip': torchvision.transforms.RandomVerticalFlip(p=0.5),
                  'hflip': torchvision.transforms.RandomHorizontalFlip(p=0.5),
                  'none': None,
}

transforms = torchvision.transforms.Compose([transform_dict['hue'], transform_dict['translate']])
cmnist_train = CMNISTWithTransform(transforms, train=True)
cmnist_val = CMNISTWithTransform(None, train=False)
train_loader = torch.utils.data.DataLoader(cmnist_train,
                                           batch_size=128,
                                           shuffle=True,
                                           **kwargs)
val_loader = torch.utils.data.DataLoader(cmnist_val,
                                         batch_size=128,
                                         shuffle=True,
                                         **kwargs)

# Generate test set
mnist = datasets.MNIST('~/datasets/mnist', train=False, download=True)
test_env = make_environment(mnist.test_data, mnist.targets, 0.9)
env2 = TensorDataset(test_env['images'], test_env['labels'], torch.zeros_like(test_env['labels']))
test_loader = torch.utils.data.DataLoader(env2,
                                         batch_size=128,
                                         shuffle=False, **kwargs)

# Init model and optimizer
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

model_name = 'cmnist_with_chosen_da_seed_' + str(args.seed)

best_val_acc = 0.
for epoch in range(1, args.epochs + 1):
    print('\n Epoch: ' + str(epoch))
    train(args, model, device, train_loader, optimizer, epoch)
    train_loss, train_acc = test(args, model, device, train_loader)
    val_loss, val_acc = test(args, model, device, val_loader)

    print(epoch, train_loss, train_acc, val_loss, val_acc)

    # Save best
    if val_acc >= best_val_acc:
        best_val_acc = val_acc

        torch.save(model, model_name + '.model')
        torch.save(args, model_name + '.config')

model = torch.load(model_name + '.model').to(device)
test_loss, test_acc = test(args, model, device, val_loader)

with open(model_name + '.txt', "w") as text_file:
    text_file.write("Test Acc: " + str(test_acc))