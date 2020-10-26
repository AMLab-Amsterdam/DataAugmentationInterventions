# Experiments in the paper have 200 epochs and no normalization

import sys
import wandb

sys.path.insert(0, "../../")

import argparse
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision
import PIL

import numpy as np

from paper_experiments.PACS.model_alexnet_caffe import caffenet
from paper_experiments.PACS.pacs_data_loader_data_augmentation import PacsDataDataAug
from paper_experiments.PACS.pacs_data_loader_norm import PacsData


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_batch = 0

    for batch_idx, (data, _, domain) in enumerate(train_loader):
        data, domain = data.to(device), domain.to(device)
        _, domain = domain.max(dim=1)

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, domain)
        loss.backward()
        optimizer.step()

        loss_batch += loss

    return loss_batch


def test(args, model, device, test_loader, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _, domain in test_loader:
            data, domain = data.to(device), domain.to(device)
            _, domain = domain.max(dim=1)

            output = model(data)
            test_loss += torch.nn.CrossEntropyLoss(reduction='mean')(output, domain) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(domain.view_as(pred)).sum().item()

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
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--test_domain', type=list, default=['photo'],
                        help='domain used during test')
    parser.add_argument('--all_domains', type=list, default=['art_painting', 'cartoon', 'photo', 'sketch'],
                        help='domain used during train')
    parser.add_argument('--da', type=str, default='brightness', choices=['brightness',
                                                                    'contrast',
                                                                    'saturation',
                                                                    'hue',
                                                                    'rotation',
                                                                    'translate',
                                                                    'scale',
                                                                    'shear',
                                                                    'vflip',
                                                                    'hflip',
                                                                    'none'
                                                                    ])

    parser.add_argument('-dd', '--data_dir', type=str, default='./data', help='Directory to download data to and load data from')
    parser.add_argument('-wd', '--wandb_dir', type=str, default='./', help='(OVERRIDDEN BY ENV_VAR for sweep) Directory to download data to and load data from')

    args = parser.parse_args()
    args.test_domain = [''.join(args.test_domain)]

    # Default args is above, Overridden by ENV_VARIABLES!!! or command line
    # Sweep interacts weirdly with some things...
    wandb.init(project="NewPACS_" + args.test_domain[0] + '_seed_' + str(args.seed), config=args, name=args.da)

    print(args)
    print("Data from:", args.data_dir)
    print("Logging to:", args.wandb_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device("cuda")

    model_name = 'pacs_' + args.da + '_seed_' + str(args.seed) + '_test_domain_' + args.test_domain[0]

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    train_domain = [n for n in args.all_domains if n != args.test_domain[0]]
    print(train_domain, args.test_domain)

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

    transforms_pacs_train = transforms.Compose([
        transform_dict[args.da],
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # transforms_pacs_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Hardcoded in the data loader

    # Load supervised training
    train_loader = data_utils.DataLoader(
        PacsDataDataAug('./kfold/', domain_list=train_domain, mode='train', transform=transforms_pacs_train),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    val_loader = data_utils.DataLoader(
        PacsData('./kfold/', domain_list=train_domain, mode='val'),
        batch_size=args.batch_size,
        shuffle=False, **kwargs)

    model = caffenet(7).to(device)

    optimizer = optim.SGD(model.parameters(), weight_decay=.0005, momentum=.9, nesterov=True, lr=args.lr)
    step_size = int(args.epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    train_accs = []
    val_accs = []
    for epoch in range(1, args.epochs + 1):
        # print('\n Epoch: ' + str(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        _, train_acc = test(args, model, device, train_loader, set_name='Train')
        val_loss, val_acc = test(args, model, device, val_loader, set_name='Val')
        wandb.log({'train accuracy': train_acc, 'val accuracy': val_acc})
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

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