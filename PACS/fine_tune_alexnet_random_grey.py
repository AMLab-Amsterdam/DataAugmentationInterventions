# Experiments in the paper have 200 epochs and no normalization

import sys
import wandb

sys.path.insert(0, "../../")

import argparse
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import numpy as np

from paper_experiments.PACS.model_alexnet_caffe import caffenet
from paper_experiments.PACS.pacs_data_loader_data_augmentation import PacsDataDataAug
from paper_experiments.PACS.pacs_data_loader import PacsData


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_batch = 0

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        _, target = target.max(dim=1)

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss.backward()
        optimizer.step()

        loss_batch += loss

    return loss_batch


def test(args, model, device, test_loader, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)

            output = model(data)
            test_loss += torch.nn.CrossEntropyLoss(reduction='mean')(output, target) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

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
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--test_domain', type=list, default=['sketch'],
                        help='domain used during test')
    parser.add_argument('--all_domains', type=list, default=['art_painting', 'cartoon', 'photo', 'sketch'],
                        help='domain used during train')

    parser.add_argument('-dd', '--data_dir', type=str, default='./data', help='Directory to download data to and load data from')
    parser.add_argument('-wd', '--wandb_dir', type=str, default='./', help='(OVERRIDDEN BY ENV_VAR for sweep) Directory to download data to and load data from')

    args = parser.parse_args()
    args.test_domain = [''.join(args.test_domain)]

    # Default config is above, Overridden by ENV_VARIABLES!!! or command line
    # Sweep interacts weirdly with some things...
    wandb.init(project="CaffeNetValLossRandomGrey_" + args.test_domain[0], config=args)

    # wandb.config.seed = args.seed
    # wandb.config.lr = args.lr
    # wandb.config.test_domain = args.test_domain

    config = wandb.config

    print(config)
    print("Data from:", config.data_dir)
    print("Logging to:", config.wandb_dir)

    use_cuda = not config.no_cuda and torch.cuda.is_available()

    # Set seed
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)

    device = torch.device("cuda")

    model_name = 'caffenet_val_loss_random_grey_seed_' + str(config.seed) + '_test_domain_' + config.test_domain[0]

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    train_domain = [n for n in config.all_domains if n != config.test_domain[0]]
    print(train_domain, config.test_domain)

    transforms_pacs_train = transforms.Compose([
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # transforms_pacs_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Hardcoded in the data loader

    # Load supervised training
    train_loader = data_utils.DataLoader(
        PacsDataDataAug('./kfold/', domain_list=train_domain, mode='train', transform=transforms_pacs_train),
        batch_size=config.batch_size,
        shuffle=True, **kwargs)
    val_loader = data_utils.DataLoader(
        PacsData('./kfold/', domain_list=train_domain, mode='val'),
        batch_size=config.batch_size,
        shuffle=False, **kwargs)

    model = caffenet(7).to(device)

    optimizer = optim.SGD(model.parameters(), weight_decay=.0005, momentum=.9, nesterov=True, lr=config.lr)
    step_size = int(config.epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    best_val_loss = 1000000000

    for epoch in range(1, config.epochs + 1):
        # print('\n Epoch: ' + str(epoch))
        train_loss = train(config, model, device, train_loader, optimizer, epoch)
        _, train_acc = test(config, model, device, train_loader, set_name='Train')
        val_loss, val_acc = test(config, model, device, val_loader, set_name='Val')

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc})

        scheduler.step()

        # Save best
        if val_loss <= best_val_loss:
            best_val_loss = val_loss

            torch.save(model, model_name + '.model')
            torch.save(args, model_name + '.config')

    # Test
    test_loader = data_utils.DataLoader(
        PacsData('./kfold/', domain_list=config.test_domain, mode='test'),
        batch_size=config.batch_size,
        shuffle=False, **kwargs)
    model = torch.load(model_name + '.model').to(device)
    _, test_acc = test(config, model, device, test_loader, set_name='Test')

    with open(model_name + '.txt', "w") as text_file:
        text_file.write("Test Acc: " + str(test_acc))


if __name__ == '__main__':
    main()