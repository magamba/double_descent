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

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets as torch_dsets, transforms as tx
from typing import Tuple, Union, List, Optional
from types import SimpleNamespace
import logging
from dataclasses import dataclass

from core.utils import DatasetInfo

logger = logging.getLogger(__name__)

DATASETS = ("cifar10", "cifar100")
Datasets = SimpleNamespace(**{ds: ds for ds in DATASETS})

DATASET_INFO_MAP = {
    "cifar10": DatasetInfo("cifar10", (3, 32, 32), 10),
    "cifar100": DatasetInfo("cifar100", (3, 32, 32), 100),
}

DatasetInfos = SimpleNamespace(**DATASET_INFO_MAP)


def with_indices(datasetclass):
    """ Wraps a DataSet class, so that it returns (data, target, index, ground_truth).
    """
    def __getitem__(self, index):
        data, target = datasetclass.__getitem__(self, index)
        try:
            ground_truth = self._targets_orig[index]
        except AttributeError:
            ground_truth = target
        
        return data, target, index, ground_truth
        
    return type(datasetclass.__name__, (datasetclass,), {
        '__getitem__': __getitem__,
    })


""" Transforms
"""

class ReshapeTransform:
    """ Assumes input of shape [..., C, H, W] and returns output
        of shape @output_shape. If output_shape is None, return output
        of shape [B, C, H, W], where B are the flattened extra dimensions
    """
    def __init__(self, output_shape=None):
        self.output_shape = tuple(output_shape) if output_shape is not None else None
    
    def __call__(self, x):
        if self.output_shape is None:
            output_shape = (-1,) + tuple(x.shape[-3:])
        else:
            output_shape = self.output_shape
        return x.view(output_shape)


class SVDTransform:
    """ Augments the given batch of images with the provided strengths. The augmentation
        calculates the svd of each image component, removes the smallest eigenvalues,
        and then reconstructs the image components again.
    
        @retun augmented_image: torch.Tensor of shape [K, A, C, H, W], denoting the K Monte Carlo directions,
                                for each of which A=len(strength) augmentations are generated. If A=1,
                                the returned tensor has shape [K, C, H, W].

    """
    def __init__(self, input_shape, mean=None, std=None, num_directions=10, strengths=[4], clip=None):
        """Set up transformation parameters
        @param input_shape: tuple shape input images (C, H, W)
        @param mean: list list of per-channel means
        @param std: list of per-channel stds
        @param num_directions: int number of augmentations to generate for each base image
        @param clip: float if not None, clip each augmentation so that its distance to the base image is clip
        @param strengths: List[int] the number of singular values to zero out, starting from the lowest
        Note: the affine translation used on PIL images by default aligns the translation
              to the nearest pixel
        """
        # Verify strength input
        assert type(strengths) == list or type(strengths) == int
        self.strengths = [strengths] if type(strengths) == int else strengths
        self.num_strengths = len(strengths)
        assert self.num_strengths > 0
        assert self.strengths == sorted(self.strengths)
        
        self.mean = mean
        self.std = std
        self.normalize = mean is not None and std is not None
        self.directions = num_directions
        self.clip = clip if clip is not None else 1.
        self._gen_geodesic_paths = self.num_strengths > 1
        if self._gen_geodesic_paths:
            self.output_shape = torch.Size([num_directions, self.num_strengths +1] + [_ for _ in input_shape])
        else:
            self.output_shape = torch.Size([num_directions +1] + [_ for _ in input_shape])

    def __call__(self, x):
        """
        @param x: PIL Image to be transformed
        @return torch.Tensor : of shape [K, C, H, W], where K=len(self.strength)
                               is the number of different augmentation strengths
        """
        x = tx.functional.to_tensor(x)
        output = torch.zeros(self.output_shape)
        if self._gen_geodesic_paths:
            output[:,0] = x
        else:
            output[0] = x
        U, S, V = torch.svd(x, some=False)
        S_orig = S.clone()
        l = S.shape[1]
        for i in range(self.directions):
            for a, s in enumerate(self.strengths):
                e = l -i # sliding window end index
                ss = s + i # sliding window start index
                S[:, -ss:e] = 0.0
            
                if self._gen_geodesic_paths:
                    output[i, a+1, :, :, :] = U @ (self.clip * torch.diag_embed(S)) @ torch.transpose(V, 1, 2)
                else:
                    output[i+1, :, :, :] = U @ (self.clip * torch.diag_embed(S)) @ torch.transpose(V, 1, 2)
                S[:, -ss:] = S_orig[:, -ss:] # restore S
        
        if self.normalize:
            output = tx.functional.normalize(
                output.view(
                    (-1,) + tuple(self.output_shape[-3:])
                ), mean=self.mean, std=self.std
            ).view(tuple(self.output_shape))

        return output
        

class ShiftTransform:
    """Shifts image by a specified radius

    For a torchvision.datasets.vision.VisionDataset, we provide a transform that:
        - Upscales image @x (copying the values at the edges)
        - Copies the image K times
        - Applies a translation to each of the K images
        - Crops the image back to its original size

    @retun transformed_x: torch.Tensor of shape [K, C, H, W], denoting the K=num_directions
                 augmented images.

    Note: the returned label is a scalar Tensor, which holds for all K
          images, so that batches of N paths correctly contain N labels.

          If needed, target_transform can be used to replicate a label
          K times.

    This way, a torch.utils.data.DataLoader can return batches of
    [N, K, C, H, W] images.
    """

    def __init__(self, input_shape, mean=None, std=None, num_directions=10, strengths=[4], clip=None):
        """Set up transformation parameters
        @param input_shape: tuple shape input images (C, H, W)
        @param mean: list list of per-channel means
        @param std: list of per-channel stds
        @param num_directions: int number of augmentations to generate for each base image (MC directions)
        @param clip: float if not None, clip each augmentation so that its distance to the base image is clip
        @param strengths: List[float] (list of) radii of the rotation used to generate the augmentations
                          If only one value is specified, it will be used to generate num_directions augmentations
                          Otherwise, A augmentations 1, .., a, .., len(strengths) will be generated, one per value
                          in strengths.
        Note: the affine translation used on PIL images by default aligns the translation
              to the nearest pixel
        """
        num_strengths = len(list(strengths))
        assert num_strengths > 0
        
        self.mean = mean
        self.std = std
        self.normalize = mean is not None and std is not None
        self.directions = num_directions
        self.clip = clip if clip is not None else 1.
        self.strengths = strengths
        self._gen_geodesic_paths = num_strengths > 1
        if self._gen_geodesic_paths:
            self.output_shape = torch.Size([num_directions, num_strengths +1] + [_ for _ in input_shape]) 
        else:
            self.output_shape = torch.Size([num_directions +1] + [_ for _ in input_shape]) 
        self.ts = np.linspace(0, 2 * np.pi, self.directions, endpoint=False)

    def __call__(self, x):
        """
        @param x: PIL Image to be transformed
        @return torch.Tensor : of shape [K, A, C, H, W], where K is the number of augmentations
                              to be generated for each strength, and A=len(self.strengths)
                              
                              If only one strength is specified, the output shape is [K, C, H, W]
                              and each sample is transformed with the same strength.
        """
        output = torch.zeros(self.output_shape)
        if self._gen_geodesic_paths:
            output[:, 0] = tx.functional.to_tensor(x)
        else:
            output[0] = tx.functional.to_tensor(x)
            
        for k, t in enumerate(self.ts): # loop over MC directions
            for a, radius in enumerate(self.strengths):
                x = tx.functional.pad(x, padding=2 * radius, padding_mode="edge")
                translated_x = tx.functional.affine(
                    x,
                    angle=0,
                    translate=(self.clip * radius * np.cos(t), self.clip * radius * np.sin(t)),
                    scale=1,
                    shear=0,
                )
                translated_x = tx.functional.center_crop(
                    translated_x, output_size=list(self.output_shape[-2:])
                )
                if self._gen_geodesic_paths:
                    output[k, a+1] = tx.functional.to_tensor(translated_x)
                else:
                    output[k+1] = tx.functional.to_tensor(translated_x)

        if self.normalize:
            output = tx.functional.normalize(
                output.view(
                    (-1,) + tuple(self.output_shape[-3:])), mean=self.mean, std=self.std
                ).view(tuple(self.output_shape))
        return output


class PathTransform:
    """Generate paths close to the support of the data distribution
       with @num_anchor anchor points from a sample @x

    For a torchvision.datasets.vision.VisionDataset, we provide a transform that:
        - Upscales image @x (copying the values at the edges)
        - Copies the image K times
        - Applies a translation to each of the K images
        - Crops the image back to its original size

    @retun path: torch.Tensor of shape [K, C, H, W], denoting the K=num_anchor
                 augmented images that form a path.

    Note: the returned label is a scalar Tensor, which holds for all K
          images, so that batches of N paths correctly contain N labels.

          If needed, target_transform can be passed to replicate a label
          K times.

    This way, a torch.utils.data.DataLoader can return batches of
    [N, K, C, H, W] images.

    Note: this design allows to iterate over a whole Dataset and generate one
          path per image within the Dataset. In order to restrict the number of
          paths returned, one should provide a sampler argument to the
          DataLoader. This is also useful for making sure that the paths
          considered contain samples with noisy and clean labels for instance.
    """

    def __init__(self, input_shape, mean=None, std=None, num_anchor=10, radius=4):
        """Set up path generation
        @param input_shape: tuple shape input images (C, H, W)
        @param mean: list list of per-channel means
        @param std: list of per-channel stds
        @param num_anchor: int number of anchor points to be generated for each path
        @param radius: float radius of the rotation used to generate the anchor points
        Note: the affine translation used on PIL images by default aligns the translation
              to the nearest pixel
        """
        self.mean = mean
        self.std = std
        self.normalize = mean is not None and std is not None
        self.num_anchor = num_anchor
        self.radius = radius
        self.output_shape = torch.Size([num_anchor] + [_ for _ in input_shape])
        self.ts = np.linspace(0, 2 * np.pi, num_anchor, endpoint=False)

    def __call__(self, x):
        """
        @param x: PIL Image to be transformed
        @return torch.Tensor : of shape [K, C, H, W], where K=@self.num_points
                               is the number of anchor points in each path
        """
        output = torch.zeros(self.output_shape)
        x = tx.functional.pad(x, padding=2 * self.radius, padding_mode="edge")

        for k, t in enumerate(self.ts):
            translated_x = tx.functional.affine(
                x,
                angle=0,
                translate=(self.radius * np.cos(t), self.radius * np.sin(t)),
                scale=1,
                shear=0,
            )
            translated_x = tx.functional.center_crop(
                translated_x, output_size=list(self.output_shape[2:])
            )
            output[k] = tx.functional.to_tensor(translated_x)

            if self.normalize:
                output = tx.normalize(output, mean=self.mean, std=self.std)
        return output.type(torch.get_default_dtype())


class MultiColorJitter(tx.ColorJitter):
    """ For input of shape [..., 1 or 3, H, W], generate
        images of the form [...., NDIR, 1 or 3, H, W] where NDIR is
        the parameter @num_directions.
        
        The different random augmentation is applied to each input [i, :],
        with controlled randomness for reproducibility.
    """
    def __init__(self, brightness=0.01, contrast=0.01, saturation=0.01, hue=0., num_directions=4, seed=44):
        super(MultiColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_directions = num_directions
        
    def _get_params(
        self,
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """ Overloads ColorJitter implementation to fix reproducible pseudorandom stream
            without affecting pytorch's randomness.
        """
        ndirs = self.num_directions
        fn_idx = [ self.rng.permutation(4) for i in range(ndirs) ]
        b = None if brightness is None else torch.from_numpy(self.rng.uniform(brightness[0], brightness[1], ndirs))
        c = None if contrast is None else torch.from_numpy(self.rng.uniform(contrast[0], contrast[1], ndirs))
        s = None if saturation is None else torch.from_numpy(self.rng.uniform(saturation[0], saturation[1], ndirs))
        h = None if hue is None else torch.from_numpy(self.rng.uniform(hue[0], hue[1], ndirs))

        return fn_idx, b, c, s, h
        
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self._get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        
        #img_c = tx.functional.to_tensor(img)
        output = [img]
        for i in range(self.num_directions):
            img_c = img.clone()
            for fn_id in fn_idx[i]:
                if fn_id == 0 and brightness_factor is not None:
                    img_c = tx.functional.adjust_brightness(img_c, brightness_factor[i])
                elif fn_id == 1 and contrast_factor is not None:
                    img_c = tx.functional.adjust_contrast(img_c, contrast_factor[i])
                elif fn_id == 2 and saturation_factor is not None:
                    img_c = tx.functional.adjust_saturation(img_c, saturation_factor[i])
                elif fn_id == 3 and hue_factor is not None:
                    img_c = tx.functional.adjust_hue(img_c, hue_factor[i])
            output.append(img_c)
        return torch.stack(output, dim=-4)
    
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue}"
            f", num_directions={self.num_directions}"
            f", seed = {self.seed})"
        )
        return s


""" Training transforms
"""
def _create_transforms(normalize, mean, std, **kwargs):
    crop_size = kwargs.pop("crop_size", 0)
    padding = kwargs.pop("padding", 4)
    hflip = kwargs.pop("hflip", False)
    crop_flipped_only = kwargs.pop("crop_flipped_only", False)
    tangent = kwargs.pop("tangent", False)
    batch_reshape_dims = kwargs.pop("batch_reshape_dims", None)

    transform_funcs = []
    if crop_size > 0:
        if crop_flipped_only:
            transform_funcs = [
                tx.RandomApply(
                    [
                        tx.RandomHorizontalFlip(p=1.),
                        tx.RandomCrop(crop_size, padding=padding),
                    ],
                    p=0.5
                )
            ]
        else:
            transform_funcs.append(tx.RandomCrop(crop_size, padding=padding))
    if hflip and not crop_flipped_only:
        transform_funcs.append(tx.RandomHorizontalFlip(p=0.5))

    transform_funcs.append(tx.ToTensor())
    if tangent:
        transform_funcs.append(MultiColorJitter())
        if normalize:
            assert batch_reshape_dims is not None
            transform_funcs += [ReshapeTransform(), tx.Normalize(mean, std), ReshapeTransform(batch_reshape_dims)]
    elif normalize:
        transform_funcs.append(tx.Normalize(mean, std))
    return tx.Compose(transform_funcs)


def _infer_batch_size(**kwargs):
    try:
        tangent = kwargs["tangent"]
    except KeyError:
        tangent = False

    try:
        num_directions = kwargs["num_directions"]
    except KeyError:
        num_directions = 0
        
    try:
        strengths = kwargs["strengths"]
        num_anchors = len(strengths)
    except KeyError:
        num_anchors = 0
    
    output_shape = ()
        
    if num_anchors > 1 :
        assert num_directions > 0
        output_shape += (num_directions, num_anchors +1)
    elif num_directions > 0:
        output_shape += (num_directions +1, )
        
    if tangent:
        output_shape += (5,)
    
    return output_shape


""" Noisy labels
"""

def corrupt_labels(dset, noise_percent, seed=None):
    from numpy.random import default_rng
    rng = default_rng(seed)
    num_labels_to_corrupt = int(round(len(dset) * noise_percent))
    if num_labels_to_corrupt == 0:
        return
    if isinstance(dset, Subset):
        all_targets = dset.dataset.targets
        dset.dataset._targets_orig = dset.dataset.targets.copy()
        if isinstance(all_targets, list):
            all_targets = torch.tensor(all_targets)
        targets = all_targets[dset.indices]
    else:
        targets = dset.targets
        dset._targets_orig = dset.targets.copy()
        if isinstance(targets, list):
            targets = torch.tensor(targets)

    num_classes = targets.unique().shape[0]

    noise = torch.zeros_like(targets)
    if num_classes == 2:
        noise[0:num_labels_to_corrupt] = 1
    else:
        noise[0:num_labels_to_corrupt] = torch.from_numpy(
            rng.integers(1, num_classes, (num_labels_to_corrupt,))
        )
    shuffle = torch.from_numpy(rng.permutation(noise.shape[0]))
    noise = noise[shuffle]
    if isinstance(dset, Subset):
        all_noisy_targets = (targets + noise) % num_classes
        if isinstance(dset.dataset.targets, list):
            for idx, noisy_label in enumerate(all_noisy_targets.tolist()):
                dset.dataset.targets[dset.indices[idx]] = noisy_label
        else:
            dset.dataset.targets[dset.indices] = all_noisy_targets
    else:
        dset.targets = (targets + noise) % num_classes


def corrupt_labels_asymmetric_balanced(dset, noise_percent, seed=None):
    """ get number of classes that we need to corrupt
        fixed corruption to 50%
    """
    assert 0. <= noise_percent <= 1., "Error: noise ratio should be a float between 0 and 1."
    from numpy.random import default_rng
    rng = default_rng(seed)
    if noise_percent == 0.:
        return
    num_labels_to_corrupt = int(round(len(dset) * noise_percent))
        
    if isinstance(dset, Subset):
        all_targets = dset.dataset.targets
        dset.dataset._targets_orig = dset.dataset.targets.copy()
        if isinstance(all_targets, list):
            all_targets = torch.tensor(all_targets)
        targets = all_targets[dset.indices]
    else:
        targets = dset.targets
        dset._targets_orig = dset.targets.copy()
        if isinstance(targets, list):
            targets = torch.tensor(targets)

    targets_orig = targets.clone()
    num_classes = targets.unique().shape[0]
    samples_per_class = len(dset) // num_classes # assuming balanced dataset
    num_pairs_to_corrupt = num_labels_to_corrupt // samples_per_class
    corrupt_idx = samples_per_class // 2
    
    for i in range(num_classes_to_corrupt):
        shuffle = torch.from_numpy(rng.permutation(samples_per_class))
        noise_first_class = np.zeros(samples_per_class)
        noise_second_class= np.zeros_like(noise_first_class)
        noise_first_class[:corrupt_idx] = 1
        noise_second_class[:corrupt_idx] = -1
        targets[targets_orig == 2*i] += torch.from_numpy(noise_first_class[shuffle]).long()
        shuffle = torch.from_numpy(rng.permutation(samples_per_class))
        targets[targets_orig == (2*i +1) % num_classes] += torch.from_numpy(noise_second_class[shuffle]).long()
    
    if isinstance(dset, Subset):
        if isinstance(dset.dataset.targets, list):
            for idx, noisy_label in enumerate(targets.tolist()):
                dset.dataset.targets[dset.indices[idx]] = noisy_label
        else:
            dset.dataset.targets[dset.indices] = targets
    else:
        dset.targets = targets
        

def corrupt_labels_asymmetric(dset, noise_percent, seed=None):
    """ get number of classes that we need to corrupt
        fixed corruption to 80%
    """
    assert 0. <= noise_percent <= 1., "Error: noise ratio should be a float between 0 and 1."
    from numpy.random import default_rng
    rng = default_rng(seed)
    if noise_percent == 0.:
        return
    num_labels_to_corrupt = int(round(len(dset) * noise_percent))
        
    if isinstance(dset, Subset):
        all_targets = dset.dataset.targets
        dset.dataset._targets_orig = dset.dataset.targets.copy()
        if isinstance(all_targets, list):
            all_targets = torch.tensor(all_targets)
        targets = all_targets[dset.indices]
    else:
        targets = dset.targets
        dset._targets_orig = dset.targets.copy()
        if isinstance(targets, list):
            targets = torch.tensor(targets)

    targets_orig = targets.clone()
    num_classes = targets.unique().shape[0]
    samples_per_class = len(dset) // num_classes # assuming balanced dataset
    num_classes_to_corrupt = num_labels_to_corrupt // int(0.8 * samples_per_class)
    corrupt_idx = int(samples_per_class * 0.8)
    
    for i in range(num_classes_to_corrupt):
        shuffle = torch.from_numpy(rng.permutation(samples_per_class))
        noise = np.zeros(samples_per_class)
        noise[:corrupt_idx] = rng.integers(1, num_classes, (corrupt_idx,))
        targets[targets_orig == i] = (targets[targets_orig == i] + torch.from_numpy(noise[shuffle]).long()) % num_classes
    
    if isinstance(dset, Subset):
        if isinstance(dset.dataset.targets, list):
            for idx, noisy_label in enumerate(targets.tolist()):
                dset.dataset.targets[dset.indices[idx]] = noisy_label
        else:
            dset.dataset.targets[dset.indices] = targets
    else:
        dset.targets = targets
        


""" Dataset interface
"""

def create_dataset(
    args,
    train=True,
    normalize=False,
    augment=False,
    subset_pct=None,
    validation=False,
    override_dset_class=None,
    **kwargs
):
    if args.data not in DATASETS:
        raise ValueError("{} is not a valid dataset".format(args.data))
        
    if args.crop_flipped_only:
        if not args.augmentation:
            raise ValueError("Error: option --crop-flipped-only requires --augmentation to be set.")

    dset = None
    try:
        tangent = kwargs["tangent"]
    except KeyError:
        tangent = False
    tangent_tx = [ MultiColorJitter() ] if tangent else []
    
    if args.data == Datasets.cifar10:
        if override_dset_class is not None:
            CIFAR10 = override_dset_class(torch_dsets.CIFAR10)
        else:
            CIFAR10 = torch_dsets.CIFAR10
        
        batch_reshape_dims = _infer_batch_size(**kwargs) + DatasetInfos.cifar10.input_shape
        kwargs["batch_reshape_dims"] = batch_reshape_dims
        gen_strategy = kwargs.pop("strategy", None)
        
        if tangent and normalize:
            tangent_tx += [
                ReshapeTransform(), tx.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)), ReshapeTransform(batch_reshape_dims)
            ]
        if gen_strategy == "shifts":
            transforms = tx.Compose(
                [
                    ShiftTransform(
                        input_shape = DatasetInfos.cifar10.input_shape,
                        mean=(0.4914, 0.4822, 0.4465) if normalize and not tangent else None,
                        std=(0.2023, 0.1994, 0.2010) if normalize and not tangent else None, 
                        num_directions=kwargs.pop("num_directions"), 
                        strengths=kwargs.pop("strengths"), 
                        clip=kwargs.pop("clip")
                    ),
                ] + tangent_tx
            )
            CIFAR10 = with_indices(CIFAR10)
        elif gen_strategy == "svd":
            transforms = tx.Compose(
                [
                    SVDTransform(
                        input_shape = DatasetInfos.cifar10.input_shape,
                        mean=(0.4914, 0.4822, 0.4465) if normalize and not tangent else None,
                        std=(0.2023, 0.1994, 0.2010) if normalize and not tangent else None, 
                        num_directions=kwargs.pop("num_directions"), 
                        strengths=kwargs.pop("strengths"), 
                        clip=kwargs.pop("clip")
                    )
                ] + tangent_tx
            )
            CIFAR10 = with_indices(CIFAR10)
        elif train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                crop_size=32,
                hflip=True,
                **kwargs
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
                **kwargs
            )
        dset = CIFAR10(
            args.data_dir, transform=transforms, train=train, download=False
        )
    elif args.data == Datasets.cifar100:
        if override_dset_class is not None:
            CIFAR100 = override_dset_class(torch_dsets.CIFAR100)
        else:
            CIFAR100 = torch_dsets.CIFAR100
        
        batch_reshape_dims = _infer_batch_size(**kwargs) + DatasetInfos.cifar100.input_shape
        kwargs["batch_reshape_dims"] = batch_reshape_dims
        gen_strategy = kwargs.pop("strategy", None)
        
        if tangent and normalize:
            tangent_tx += [
                ReshapeTransform(), tx.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2009, 0.1984, 0.2023)), ReshapeTransform(batch_reshape_dims)
            ]
        
        if gen_strategy == "shifts":
            transforms = tx.Compose(
                [
                    ShiftTransform(
                        input_shape = DatasetInfos.cifar100.input_shape,
                        mean=(0.5071, 0.4865, 0.4409) if normalize and not tangent else None,
                        std=(0.2009, 0.1984, 0.2023) if normalize and not tangent else None, 
                        num_directions=kwargs.pop("num_directions"), 
                        strengths=kwargs.pop("strengths"), 
                        clip=kwargs.pop("clip")
                    ),
                ] + tangent_tx
            )
            CIFAR100 = with_indices(CIFAR100)
        elif gen_strategy == "svd":
            transforms = tx.Compose(
                [
                    SVDTransform(
                        input_shape = DatasetInfos.cifar100.input_shape,
                        mean=(0.5071, 0.4865, 0.4409) if normalize and not tangent else None,
                        std=(0.2009, 0.1984, 0.2023) if normalize and not tangent else None, 
                        num_directions=kwargs.pop("num_directions"), 
                        strengths=kwargs.pop("strengths"), 
                        clip=kwargs.pop("clip")
                    ),
                ] + tangent_tx
            )
            CIFAR100 = with_indices(CIFAR100)
        elif train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2009, 0.1984, 0.2023),
                crop_size=32,
                hflip=True,
                **kwargs
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.5071, 0.4865, 0.4409),
                (0.2009, 0.1984, 0.2023),
                **kwargs
            )
        dset = CIFAR100(
            args.data_dir, transform=transforms, train=train, download=False
        )
    if subset_pct is not None and 1 > subset_pct > 0:
        rng = np.random.default_rng(args.data_split_seed)
        shuffle = rng.permutation(len(dset))
        split_index = int(subset_pct * len(dset))
        if validation:
            rand_indices = torch.from_numpy(shuffle)[split_index:]
        else:
            rand_indices = torch.from_numpy(shuffle)[:split_index]
        dset = Subset(dset, rand_indices)
    return dset


""" Data loaders
"""

@dataclass
class DataManager:
    dset: Dataset
    dloader: DataLoader
    tloader: DataLoader
    vloader: DataLoader
    vset: Dataset
    tset: Dataset


def create_data_manager(
    args,
    noise,
    seed=None,
    normalize=True,
    augment=False,
    train_validation_split=(None, None),
    train_subset_pct=None,
    test_subset_pct=None,
    override_dset_class=None,
    **kwargs
):
    dset = create_dataset(
        args,
        train=True,
        normalize=normalize,
        augment=augment,
        subset_pct=train_subset_pct,
        override_dset_class=override_dset_class,
        **kwargs
    )
    tset = create_dataset(
        args,
        train=False,
        normalize=normalize,
        augment=False,
        subset_pct=train_subset_pct,
        override_dset_class=override_dset_class,
        **kwargs
    )
    vset, vloader = None, None
    if train_validation_split != (None, None):
        vset = create_dataset(
            args,
            train=True,
            normalize=normalize,
            augment=False,
            subset_pct=train_subset_pct,
            validation=True,
            override_dset_class=override_dset_class,
            **kwargs
        )
        if seed is not None:
            torch.manual_seed(seed)
        rng_state = torch.get_rng_state()
        logger.info("Splitting training set into train: {}, val: {}.".format(
                train_validation_split[0], train_validation_split[1]
            )
        )
        _, vset = random_split(
            vset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        dset, _ = random_split(
            dset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        torch.set_rng_state(rng_state)
    if args.noise_type == "symmetric":
        corrupt_labels(dset, noise, args.label_seed)
    else:
        corrupt_labels_asymmetric(dset, noise, args.label_seed)

    pin_memory=True
    try:
        shuffle = args.gen_strategy is None
    except AttributeError:
        shuffle = True
    
    logger.info("Running with {} cpu workers.".format(args.workers))
    kwargs_train = {
        "batch_size": args.batch_size,
        "num_workers" : args.workers,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
    }
    kwargs_no_train = {
        "batch_size": args.batch_size,
        "num_workers" : args.workers,
        "shuffle": False,
        "pin_memory": pin_memory,
    }
    
    if train_validation_split != (None, None):
        vloader = DataLoader(vset, **kwargs_no_train)
    dloader = DataLoader(dset, **kwargs_train)
    tloader = DataLoader(tset, **kwargs_no_train)

    return DataManager(
        dset,
        dloader,
        tloader,
        vloader,
        vset,
        tset,
    )
