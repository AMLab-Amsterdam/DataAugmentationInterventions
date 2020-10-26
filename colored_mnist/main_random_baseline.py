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


# Define and instantiate the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        lin1 = nn.Linear(2 * 14 * 14, args.hidden_dim)
        lin2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        lin3 = nn.Linear(args.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return F.sigmoid(out)


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
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.),
        'labels': labels[:, None]}


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
args = parser.parse_args()

# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)

# Load MNIST, make train/val splits, and shuffle train set examples

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())


envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)]

# Put into data loader
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': False}

env0 = TensorDataset(envs[0]['images'], envs[0]['labels'], torch.zeros_like(envs[0]['labels']))
env1 = TensorDataset(envs[1]['images'], envs[1]['labels'], torch.zeros_like(envs[1]['labels'])+1.)
env2 = TensorDataset(envs[2]['images'], envs[2]['labels'], torch.zeros_like(envs[2]['labels']))
cmnist = torch.utils.data.ConcatDataset([env0, env1])
train_loader = torch.utils.data.DataLoader(cmnist,
                                         batch_size=100,
                                         shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(env2,
                                         batch_size=100,
                                         shuffle=False, **kwargs)

# Init model and optimizer
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, args.epochs + 1):
    print('\n Epoch: ' + str(epoch))
    train(args, model, device, train_loader, optimizer, epoch)
    train_loss, train_acc = test(args, model, device, train_loader)
    test_loss, test_acc = test(args, model, device, test_loader)

    print(epoch, train_loss, train_acc, test_loss, test_acc)

    # # Save best
    # if val_acc >= best_val_acc:
    #     best_val_acc = val_acc
    #
    #     torch.save(model, model_name + '.model')
    #     torch.save(args, model_name + '.config')
    #     early_stopping = 0
    #
    # early_stopping += 1
    #
    # if early_stopping >= args.early_stop_after:
    #     break
