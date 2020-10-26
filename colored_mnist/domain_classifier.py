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
from PIL import Image, ImageFilter
# import skimage.color
import cv2
from torchvision.utils import save_image
import torchvision

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.train_data[:50000], mnist.train_labels[:50000])
    mnist_val = (mnist.train_data[50000:], mnist.train_labels[50000:])

    mnist_test = datasets.MNIST('~/datasets/mnist', train=False, download=True)
    mnist_test = (mnist_test.test_data, mnist_test.test_labels)

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())


    # Build environments
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()


    def make_environment(images, labels, e, domain_id):
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
            'images': (images.float() / 255.).cuda(),
            'labels': labels.cuda(),
            'domain': torch.zeros_like(labels).cuda().long() + domain_id
        }

    # env1 = make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2, 0)
    # env2 = make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, 1)
    # env3 = make_environment(mnist_val[0], mnist_val[1], 0.9, 2)
    #
    # a = 1

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2, 0),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, 1),
        make_environment(mnist_val[0], mnist_val[1], 0.9, 2)
    ]

    envs_test = [
        make_environment(mnist_test[0][::3], mnist_test[1][::3], 0.2, 0),
        make_environment(mnist_test[0][1::3], mnist_test[1][1::3], 0.1, 1),
        make_environment(mnist_test[0][2::3], mnist_test[1][2::3], 0.9, 2)
    ]

    # # Randomly shuffle color channels
    # for i in range(3):
    #     shuffle_or_not = torch_bernoulli(0.5, envs[i]['images'].shape[0])
    #     for j in range(envs[i]['images'].shape[0]):
    #         if shuffle_or_not[j]:
    #             temp = envs[i]['images'][j][0].clone()
    #             envs[i]['images'][j][0] = envs[i]['images'][j][1]
    #             envs[i]['images'][j][1] = temp

    # # Check augmentations
    # images_zero = torch.zeros((64, 1, 14, 14))
    # save_img = envs[2]['images'][:64].cpu()
    # save_image(torch.cat([save_img, images_zero], dim=1),
    #            'color_mnist_shuffle_test.png', nrow=8)


    # Define and instantiate the model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 3)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            out = self._main(out)
            return out


    mlp = MLP().cuda()


    # Define loss function helpers
    def mean_nll(logits, y):
        return nn.CrossEntropyLoss()(logits, y)


    def mean_accuracy(logits, y):
        preds = logits.argmax(dim=1, keepdim=True)
        return preds.eq(y.view_as(preds)).float().mean()


    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)


    # Train loop
    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))


    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['domain'])
            env['acc'] = mean_accuracy(logits, env['domain'])
            # print(env['nll'], env['acc'])
            # env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll'], envs[2]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc'], envs[2]['acc']]).mean()
        # train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        # penalty_weight = (flags.penalty_weight
        #     if step >= flags.penalty_anneal_iters else 1.0)
        # loss += penalty_weight * train_penalty
        # if penalty_weight > 1.0:
        #   # Rescale the entire loss to keep gradients in a reasonable range
        #   loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for env in envs_test:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['domain'])
            env['acc'] = mean_accuracy(logits, env['domain'])

        test_acc = torch.stack([envs_test[0]['acc'], envs_test[1]['acc'], envs_test[2]['acc']]).mean()
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                # train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )
    #
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
