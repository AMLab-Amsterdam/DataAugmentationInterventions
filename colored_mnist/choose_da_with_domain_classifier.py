import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
import PIL
import wandb


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
    for batch_idx, (data, _, domain) in enumerate(train_loader):
        data, domain = data.to(device), domain.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCELoss()(output, domain)
        loss.backward()
        optimizer.step()
    #
        # if batch_idx==1:
        #     save_image(data[:8].cpu(),
        #                'brightness' + str(epoch) + '.png', nrow=8)
        #     print(_[:8], domain[:8])
        #     a = 1

    # for batch_idx, (data, target, _) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)
    #
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = nn.BCELoss()(output, target)
    #     loss.backward()
    #     optimizer.step()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _, domain in test_loader:
            data, domain = data.to(device), domain.to(device)

            output = model(data)
            test_loss += nn.BCELoss()(output, domain).item() # sum up batch loss
            pred = output >= 0.5
            correct += pred.eq(domain.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)

    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target, _ in test_loader:
    #         data, target = data.to(device), target.to(device)
    #
    #         output = model(data)
    #         test_loss += nn.BCELoss()(output, target).item()  # sum up batch loss
    #         pred = output >= 0.5
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    # test_loss /= len(test_loader.dataset)
    #
    # return test_loss, 100. * correct / len(test_loader.dataset)


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
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
parser.add_argument('-dd', '--data_dir', type=str, default='./data',
                    help='Directory to download data to and load data from')
parser.add_argument('-wd', '--wandb_dir', type=str, default='./',
                    help='(OVERRIDDEN BY ENV_VAR for sweep) Directory to download data to and load data from')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Put into data loader
device = torch.device("cuda")
kwargs = {'num_workers': 8, 'pin_memory': False}

model_name = 'cmnist_' + args.da +'_seed_' + str(args.seed)
wandb.init(project="NewNewCMNIST" + str(args.seed), config=args, name=args.da)

# Data aug
transform_dict = {'brightness': torchvision.transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
                  'contrast': torchvision.transforms.ColorJitter(brightness=0, contrast=10.0, saturation=0, hue=0),
                  'saturation': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=10.0, hue=0),
                  'hue': torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
                  'rotation': torchvision.transforms.RandomAffine([0, 359], translate=None, scale=None, shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'translate': torchvision.transforms.RandomAffine(0, translate=[0.2, 0.2], scale=None, shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'scale': torchvision.transforms.RandomAffine(0, translate=None, scale=[0.8, 1.2], shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                  'shear': torchvision.transforms.RandomAffine(0, translate=None, scale=None, shear=[-10., 10., -10., 10.], resample=PIL.Image.BILINEAR, fillcolor=0),
                  'vflip': torchvision.transforms.RandomVerticalFlip(p=0.5),
                  'hflip': torchvision.transforms.RandomHorizontalFlip(p=0.5),
                  'none': None,
}

cmnist_train = CMNISTWithTransform(transform_dict[args.da], train=True)
cmnist_val = CMNISTWithTransform(None, train=False)

train_loader = torch.utils.data.DataLoader(cmnist_train,
                                           batch_size=128,
                                           shuffle=True,
                                           **kwargs)
val_loader = torch.utils.data.DataLoader(cmnist_val,
                                         batch_size=128,
                                         shuffle=True,
                                         **kwargs)

# Init model and optimizer
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_accs = []
val_accs = []
for epoch in range(1, args.epochs + 1):
    print('\n Epoch: ' + str(epoch))
    train(args, model, device, train_loader, optimizer, epoch)
    train_loss, train_acc = test(args, model, device, train_loader)
    val_loss, val_acc = test(args, model, device, val_loader)

    print(epoch, train_loss, train_acc, val_loss, val_acc)
    wandb.log({'train acc': train_acc, 'val acc': val_acc})
    train_accs.append(train_acc)
    val_accs.append(val_acc)

train_accs = np.array(train_accs)
mean_train_accs = np.mean(train_accs[-10:])
print(mean_train_accs)

val_accs = np.array(val_accs)
mean_val_accs = np.mean(val_accs[-10:])
print(mean_val_accs)

with open(model_name + '.txt', "w") as text_file:
    text_file.write(str(mean_train_accs) + ',' + str(mean_val_accs))

wandb.run.summary["mean_train_accs"] = mean_train_accs
wandb.run.summary["mean_val_accs"] = mean_val_accs
