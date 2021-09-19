import torch.nn.functional as f
from torch.nn import init
from torch import nn


class model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = []
        in_channel = [2, 8, 16, 32]
        out_channel = [8, 16, 32,64]
        kernel_size = [(5, 5), (3, 3), (3, 3), (3, 3)]
        padding = [(2, 2), (1, 1), (1, 1), (1, 1)]

        for i in range(4):
            conv = nn.Conv2d(in_channel[i], out_channel[i], kernel_size[i], (2, 2), padding[i])
            relu = nn.ReLU()
            bn = nn.BatchNorm2d(out_channel[i])
            
            init.kaiming_normal_(conv.weight, a=0.1)
            conv.bias.data.zero_()
            
            self.layers += [conv, relu, bn]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        self.conv = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.lin(x)
        return x
