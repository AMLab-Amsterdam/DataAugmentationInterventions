import sys
sys.path.insert(0, "../../../")

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np

from paper_experiments.rotated_MNIST.mnist_loader_shifted_label_distribution_rotate import MnistRotatedDist
from paper_experiments.rotated_MNIST.mnist_loader_shifted_label_distribution_flip import MnistRotatedDistFlip

from paper_experiments.rotated_MNIST.mnist_loader import MnistRotated
from paper_experiments.rotated_MNIST.augmentations.model_baseline import Net


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        _, target = target.max(dim=1)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--da', type=str, default='rotate', choices=['rotate', 'flip'],
                        help='type of data augmentation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device("cuda")
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    # Load supervised training
    if args.da == 'rotate':
        mnist_0 = MnistRotatedDist('../dataset/', train=True, thetas=[0.0], d_label=0, transform=True)
        mnist_60 = MnistRotatedDist('../dataset/', train=True, thetas=[60.0], d_label=1, transform=True)
        mnist_90 = MnistRotatedDist('../dataset/', train=True, thetas=[90.0], d_label=2, transform=True)
        model_name = 'baseline_test_0_random_rotate_seed_' + str(args.seed)

    elif args.da == 'flip':
        mnist_0 = MnistRotatedDistFlip('../dataset/', train=True, thetas=[0.0], d_label=0)
        mnist_60 = MnistRotatedDistFlip('../dataset/', train=True, thetas=[60.0], d_label=1)
        mnist_90 = MnistRotatedDistFlip('../dataset/', train=True, thetas=[90.0], d_label=2)
        model_name = 'baseline_test_0_random_flips_seed_' + str(args.seed)

    mnist = data_utils.ConcatDataset([mnist_0, mnist_60, mnist_90])

    train_size = int(0.9 * len(mnist))
    val_size = len(mnist) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist, [train_size, val_size])

    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)

    val_loader = data_utils.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False, **kwargs)

    model = Net().to(device)
    # model = NetFlat().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        print('\n Epoch: ' + str(epoch))
        train(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = test(args, model, device, val_loader)

        print(epoch, val_loss, val_acc)

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc

            torch.save(model, model_name + '.model')
            torch.save(args, model_name + '.config')


    # Test loader
    mnist_30 = MnistRotated('../dataset/', train=False, thetas=[30.0], d_label=0)
    test_loader = data_utils.DataLoader(mnist_30,
                                        batch_size=args.batch_size,
                                        shuffle=False, **kwargs)

    model = torch.load(model_name + '.model').to(device)
    _, test_acc = test(args, model, device, test_loader)

    with open(model_name + '.txt', "w") as text_file:
        text_file.write("Test Acc: " + str(test_acc))


if __name__ == '__main__':
    main()