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

from types import SimpleNamespace

from core.data import DATASET_INFO_MAP
from models.vgg import MODEL_FACTORY_MAP as VGG_FACTORY_MAP
from models.resnet import MODEL_FACTORY_MAP as RESIDUAL_FACTORY_MAP
from models.cnn import MODEL_FACTORY_MAP as CNN_FACTORY_MAP
from models.concepts import NetworkAddition

MODEL_FACTORY_MAP = {**VGG_FACTORY_MAP, **RESIDUAL_FACTORY_MAP, **CNN_FACTORY_MAP}

Models = SimpleNamespace(**MODEL_FACTORY_MAP)
Models.names = SimpleNamespace(**{name: name for name in MODEL_FACTORY_MAP})


def create_model(model_name, data_name, additions=(), *args, **kwargs):
    if model_name not in MODEL_FACTORY_MAP:
        raise KeyError("{} is not in MODELS_MAP".format(model_name))
    if data_name not in DATASET_INFO_MAP:
        raise KeyError("{} is not in ALL_DATASETS".format(data_name))
    dropout_rate = kwargs.pop("dropout_rate", 0.)
    network_builder = MODEL_FACTORY_MAP[model_name](
        DATASET_INFO_MAP[data_name], *args, **kwargs
    )
    for addition in additions:
        network_builder.add(NetworkAddition(addition), dropout_rate=dropout_rate)
    return network_builder.build_net()
