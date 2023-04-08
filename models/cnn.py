# -*- coding: utf-8 -*-

# Taken from:
# https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/mcnn.py

## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/

from functools import partial
from models.concepts import NetworkBuilder, NetworkAddition


import torch.nn as nn
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

def make_cnn(c, num_classes, use_batch_norm):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(c) if use_batch_norm else nn.Identity(),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*2) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*4) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*8) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*8, num_classes, bias=True)
    )


class CNNBuilder(NetworkBuilder):
    def __init__(self, cnn_fn, c, dataset_info, use_batch_norm, **kwargs):
        self.build_fn = cnn_fn
        self.c = c
        self.use_bn = use_batch_norm
        self.num_classes = dataset_info.output_dimension
        super().__init__(dataset_info)

    def add(self, addition: NetworkAddition, **kwargs):
        if addition == NetworkAddition.BATCH_NORM:
            self.add_batch_norm()
        if addition == NetworkAddition.DROPOUT:
            self.add_dropout()

    def add_batch_norm(self):
        self.use_bn = True

    def add_dropout(self):
        raise NotImplementedError("Dropout for ResNet not supported yet")

    def build_net(self) -> nn.Module:
        return self.build_fn(self.c, self.num_classes, self.use_bn)


cnn_increasing_widths = {
    "cnn_" + str(width): partial(CNNBuilder, make_cnn, width) for width in range(1,65)
}

MODEL_FACTORY_MAP = {
    **cnn_increasing_widths
}

