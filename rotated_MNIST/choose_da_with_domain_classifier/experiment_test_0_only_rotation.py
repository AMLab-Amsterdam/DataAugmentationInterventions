import sys
sys.path.insert(0, "../../../")

import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import PIL
import wandb

from paper_experiments.rotated_MNIST.mnist_loader_shifted_label_distribution_rotate_da import MnistRotatedDistDa
from paper_experiments.rotated_MNIST.augmentations.model_baseline import Net


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _, domain) in enumerate(train_loader):
        data, domain = data.to(device), domain.to(device)
        _, domain = domain.max(dim=1)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, domain)
        loss.backward()
        optimizer.step()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _, domain in test_loader:
            data, domain = data.to(device), domain.to(device)
            _, domain = domain.max(dim=1)

            output = model(data)
            test_loss += F.nll_loss(output, domain, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(domain.view_as(pred)).sum().item()

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
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--da', type=str, default='rotation1', choices=['rotation1',
                                                                 'rotation2',
                                                                 'rotation3',
                                                                 'rotation4',
                                                                 'rotation5',
                                                                 ])
    parser.add_argument('-dd', '--data_dir', type=str, default='./data',
                        help='Directory to download data to and load data from')
    parser.add_argument('-wd', '--wandb_dir', type=str, default='./',
                        help='(OVERRIDDEN BY ENV_VAR for sweep) Directory to download data to and load data from')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(args.da)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda")
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    transform_dict = {'rotation1': torchvision.transforms.RandomAffine(15, translate=None, scale=None, shear=None),
                      'rotation2': torchvision.transforms.RandomAffine(45, translate=None, scale=None, shear=None),
                      'rotation3': torchvision.transforms.RandomAffine(90, translate=None, scale=None,
                                                                       shear=None),
                      'rotation4': torchvision.transforms.RandomAffine([0, 180], translate=None, scale=None,
                                                                       shear=None),
                      'rotation5': torchvision.transforms.RandomAffine([0, 359], translate=None, scale=None,
                                                                       shear=None)
                      }

    rng_state = np.random.get_state()
    mnist_0_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[0], d_label=0, transform=transform_dict[args.da],
                                       rng_state=rng_state)
    mnist_0_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[0], d_label=0, transform=None,
                                     rng_state=rng_state)
    rng_state = np.random.get_state()
    mnist_30_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[30.0], d_label=1,
                                        transform=transform_dict[args.da], rng_state=rng_state)
    mnist_30_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[30.0], d_label=1, transform=None,
                                      rng_state=rng_state)
    rng_state = np.random.get_state()
    mnist_60_train = MnistRotatedDistDa('../dataset/', train=True, thetas=[60.0], d_label=2,
                                        transform=transform_dict[args.da], rng_state=rng_state)
    mnist_60_val = MnistRotatedDistDa('../dataset/', train=False, thetas=[60.0], d_label=2,
                                      transform=None, rng_state=rng_state)
    mnist_train = data_utils.ConcatDataset([mnist_0_train, mnist_30_train, mnist_60_train])
    train_loader = data_utils.DataLoader(mnist_train,
                                         batch_size=100,
                                         shuffle=True,
                                         **kwargs)

    mnist_val = data_utils.ConcatDataset([mnist_0_val, mnist_30_val, mnist_60_val])
    val_loader = data_utils.DataLoader(mnist_val,
                                       batch_size=100,
                                       shuffle=True,
                                       **kwargs)

    wandb.init(project="NewRotated_MNISTOnlyRot", config=args, name=args.da)
    model_name = 'baseline_test_0_'+ args.da +'_seed_' + str(args.seed)


    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_accs = []
    val_accs = []
    for epoch in range(1, args.epochs + 1):
        print('\n Epoch: ' + str(epoch))
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss, train_acc = test(args, model, device, train_loader)
        val_loss, val_acc = test(args, model, device, val_loader)

        wandb.log({'train accuracy': train_acc, 'val accuracy': val_acc})
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    train_accs = np.array(train_accs)
    mean_train_accs = np.mean(train_accs[-10:])
    print(mean_train_accs)

    val_accs = np.array(val_accs)
    mean_val_accs = np.mean(val_accs[-10:])
    print(mean_val_accs)

    with open(model_name + '.txt', "w") as text_file:
        text_file.write("Mean train acc: " + str(mean_train_accs))
        text_file.write("Mean val acc: " + str(mean_val_accs))

    wandb.run.summary["mean_train_accs"] = mean_train_accs
    wandb.run.summary["mean_val_accs"] = mean_val_accs


if __name__ == '__main__':
    main()