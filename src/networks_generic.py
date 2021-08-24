import torch
import torch.nn as nn
import torch.nn.functional as F


class Generic_Net(torch.nn.Module):
    def __init__(self, layer_sizes=[256, 128, 2], dropout_prob=None, device=None):
        super(Generic_Net, self).__init__()
        self.device = device

        if dropout_prob is not None and dropout_prob > 0.5:
            print("Are you sure dropout_prob is supposed to be greater than 0.5?")

        # # Load ResNet
        # resnet_full = torch.hub.load(
        #     "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        # )
        # self.resnet = torch.nn.Sequential(*list(resnet_full.children())[:-1])
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.eval()

        # Layers
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.drops = None if dropout_prob is None else nn.ModuleList()
        prev_size = 512 
        for i, size in enumerate(layer_sizes):
            self.bns.append(nn.BatchNorm1d(prev_size))
            self.fcs.append(nn.Linear(prev_size, size))
            if dropout_prob is not None:
                self.drops.append(nn.Dropout(p=dropout_prob))
            prev_size = size

        print("RECHED GENERIC TUNING")


    def forward(self, inputs):
        image = inputs
        x = image
        zipped_layers = (
            zip(self.bns, self.fcs, [None] * len(self.bns))
            if self.drops is None
            else zip(self.bns, self.fcs, self.drops)
        )
        for i, (bn, fc, drop) in enumerate(zipped_layers):
            x = bn(x)
            if drop is not None:
                x = drop(x)
            if i == len(self.bns) - 1:
                x = fc(x)
            else:
                x = F.relu(fc(x))

        return x
