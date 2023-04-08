# -*- coding: utf-8 -*-

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>. 

from enum import Enum
import abc

import torch
import torch.nn as nn


class NetworkAddition(Enum):
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"


ALL_ADDITIONS = {NetworkAddition.BATCH_NORM.value, NetworkAddition.DROPOUT.value}


class NetworkBuilder(abc.ABC):
    def __init__(self, dataset_info):
        self._dataset_info = dataset_info

    @abc.abstractmethod
    def add(self, addition: NetworkAddition, **kwargs):
        """Add the network component addition, to the network"""

    @abc.abstractmethod
    def build_net(self) -> nn.Module:
        """
        Take whatever internal state this keeps and convert it into a module
        object to be consumed metrics
        """

