import torch
import torch.nn as nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 128),
                                        nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(128, 10))

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        h = self.classifier(h)
        output = F.log_softmax(h, dim=1)
        return output