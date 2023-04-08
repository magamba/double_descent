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

import torch
from typing import Tuple

""" Helpers for Metrics.
"""

@torch.jit.script
def accuracy(output: torch.Tensor, targets: torch.Tensor, k: int=1) -> torch.Tensor:
    """ Compute top-K 0/1 loss, without averaging along the batch dimension
    """
    repeat_factor = output.shape[0] // targets.shape[0]
    prediction = output.topk(k, dim=1)[1].squeeze()
    acc = torch.eq(prediction, targets.repeat_interleave(repeat_factor))
    return acc


""" Monte Carlo integration
"""

@torch.jit.script
def integrate(f: torch.Tensor) -> torch.Tensor:
    """ Integrate @x along the first batch dimension
        by averaging
    """
    return torch.mean(
        0.5 * (f[:,0][:,None] + f[:,1:]),
        dim=1
    )


@torch.jit.script
def integrate_line(f: torch.Tensor, path_segment_dist: torch.Tensor) -> torch.Tensor:
    """ Compute the line integral of @f along the second batch dimension, and integrate
        by averaging along the first batch dimension
    """
    # -- Integrate over all paths (trapezoidal), normalize each path, then MC integrate 
    nanchors = f.shape[2] 
    path_integral = torch.sum(0.5*(f[:,:,0:nanchors-1] + f[:,:,1:nanchors]) * path_segment_dist, dim=2) # per-path integral
    path_integral_normalized = path_integral / torch.sum(path_segment_dist, dim=2)
    expectation = torch.mean(path_integral_normalized, dim=1) # expectation over paths
    return expectation


@torch.jit.script
def integrate_line_cumulative(f: torch.Tensor, path_segment_dist: torch.Tensor) -> torch.Tensor:
    """ Compute the line integrals over the sequence of intervals [a_0, a_1] c [a_0, a_1, a_2] c ... c [a_0, ..., a_K]
        where a_i are the per-path augmentations (anchor points). Average the results (MC integration) over
        multiple MC samples.
    """
    nanchors = f.shape[2]
    path_distances_cumsum = torch.cumsum(path_segment_dist, dim=2)
    path_integral = torch.cumsum(
        0.5 * (f[:,:,0:nanchors-1] + f[:,:,1:nanchors]) * path_segment_dist,
        dim=2
    ) / torch.cumsum(path_segment_dist, dim=2)
    expectation = torch.mean(path_integral, dim=1)
    return expectation


""" Normalization and distances
"""

@torch.jit.script
def batch_normalize(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ Normalize tensor @x of shape (N, *) by dividing each entry x[i, ...]
        by its Frobenius norm.
    """
    x = x.reshape(batch_size, -1)
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / x_norm


@torch.jit.script
def path_distances(x: torch.Tensor, nsamples: int, ndirections: int, nanchors: int) -> torch.Tensor:
    """ Compute distance between consecutive points in a batched fashion.
        @param x: torch.Tensor of shape (N, K, A, *), with batch size N,
                  K MC sampling directions, and A points in a sequence 1, ..., a, ..., A.
        @param nsamples: int The N dimension
        @param ndirections: int The K dimension
        @param nanchors: int The A dimension
        @return batched distance (N, K, A, *), x[:, :, a+1] - x[:, :, a]
    """
    x_next = torch.roll(x.view(nsamples, ndirections, nanchors, -1).clone(), shifts=1, dims=2)
    return torch.norm(
        x.view(nsamples * ndirections * nanchors, -1) - x_next.view(nsamples * ndirections * nanchors, -1),
        p=2, dim=1
    ).view(nsamples, ndirections, nanchors)[:,:,1:]


def get_model_forward(model, normalization):
    """ Wrapper to select model forward pass computation algorith based on
        output normalization strategy.
    """
    device = next(model.parameters()).device
    if normalization == "logsoftmax":
        criterion = torch.nn.LogSoftmax(dim=1).to(device)
    elif normalization == "softmax":
        criterion = torch.nn.Softmax(dim=1).to(device)
    elif normalization == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = None
    
    if criterion is None:
        model_forward = model
    elif normalization == "crossentropy":
        def model_forward(x, targets):
            out = model(x)
            repeat_factor = out.shape[0] // targets.shape[0]
            return criterion(out, targets.repeat_interleave(repeat_factor))
    else:
        def model_forward(x):
            out = model(x)
            return criterion(out)

    return model_forward


""" Jacobian computation
"""

def get_jacobian_fn(model, nsamples, nclasses, bigmem=False, target_logit_only=False, normalization=None):
    """Wrapper to select Jacobian computation algorithm
    """
    scalar_output = target_logit_only or (normalization == "crossentropy")
    if target_logit_only and normalization == "crossentropy":
        raise ValueError("Unsupported combination: target logit only and crossentropy normalization.")
    elif target_logit_only:
        jacobian = jacobian_target_only
    elif normalization == "crossentropy":
        jacobian = jacobian_cross_entropy
    elif bigmem:
        jacobian = jacobian_big_mem
    else:
        jacobian = jacobian_low_mem
    
    def tile_input(x, bigmem=False):
        if bigmem and not scalar_output:
            tile_shape = (nclasses,) + (1,) * len(x.shape[1:])
            return x.repeat(tile_shape)
        else:
            return x
   
    model_forward = get_model_forward(model, normalization)

    def jacobian_fn(x, targets=None):
        x = tile_input(x, bigmem)
        x.requires_grad_(True)
        
        if targets is None:
            output = model_forward(x)
            j = jacobian(x, output, nsamples, nclasses)
        elif target_logit_only:
            output = model_forward(x)
            j = jacobian(x, output, targets, nsamples)
        else:
            output = model_forward(x, targets)
            j = jacobian(x, output, nsamples, nclasses)
        x.grad = None
        return j
    
    return jacobian_fn


@torch.jit.script
def jacobian_cross_entropy(x: torch.Tensor, predictions: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @predictions: 1D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    predictions.backward(gradient=torch.ones_like(predictions), retain_graph=True)
    jacobian = x.grad.data.view(nsamples, -1)
    
    return jacobian


@torch.jit.script
def jacobian_big_mem(x: torch.Tensor, logits: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    indexing_mask = torch.eye(nclasses, device=x.device).repeat((nsamples,1))
    
    logits.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, nclasses, -1).transpose(1,2)
    
    return jacobian


@torch.jit.script
def jacobian_low_mem(x: torch.Tensor, logits: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    jacobian = torch.zeros(
        x.shape + (nclasses,), dtype=x.dtype, device=x.device
    )
    indexing_mask = torch.zeros_like(logits)
    indexing_mask[:, 0] = 1.
    
    for dim in range(nclasses):
        logits.backward(gradient=indexing_mask, retain_graph=True)
        jacobian[..., dim] = x.grad.data
        x.grad.data.zero_()
        indexing_mask = torch.roll(indexing_mask, shifts=1, dims=1)
    
    return jacobian


@torch.jit.script
def jacobian_target_only(x: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, nsamples: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits[targets] w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            @targets: 1D Tensor: Batch of logit indices, specifying which logit
                      to use for computing the Jacobian.
            
        Return:
            Jacobian: Jacobian: Batch-indexed 2D torch.Tensor of shape (N,D).
            where N is the batch dimension, D is the (flattened) input
            space dimension.
    """
    x.retain_grad()
    jacobian = torch.zeros_like(x)
    indices = torch.arange(nsamples, device=x.device)
    indexing_mask = torch.zeros_like(logits)
    repeat_factor = logits.shape[0] // targets.shape[0]
    indexing_mask[indices, targets.repeat_interleave(repeat_factor)] = 1.
    
    logits.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, -1)
    return jacobian


""" Tangent Norm Computations
"""

def get_tangent_norm_fn(data_shape: Tuple, normalize: bool = False):
    """Select tangent norm computation function.
    """
    nbatch_dims = len(data_shape[:-3])
    assert nbatch_dims > 1, "Tangent computation at least 2 batch dimensions, but found 1."
    
    if nbatch_dims == 2:
        if normalize:
            return tangent_norm_normalized
        else:
            return tangent_norm
    if nbatch_dims == 3:
        if normalize:
            return tangent_norm_euclidean_normalized
        else:
            return tangent_norm_euclidean
    elif nbatch_dims == 4:
        if normalize:
            return tangent_norm_geodesic_normalized
        else:
            return tangent_norm_geodesic
    else:
        raise ValueError("Unsupported data shape: {}".format(data_shape))
    

@torch.jit.script
def tangent_norm(f: torch.Tensor) -> torch.Tensor:
    """Compute local similarity measure described in LeJeune et al., 2019.
        @param f batch-indexed tensor of (transposed) function of shape (N, *, Kl)
               with N batch size, * is an arbitrary number of input dimensions, Kl is the
               number of output dimensions.
    """
    nsamples = f.shape[0]
    ndirections = f.shape[1]
    f = f.reshape(nsamples, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,0].view(nsamples, 1, -1) - f[:,1:].view(nsamples, ndirections -1, -1)) / 0.1,
            p=2, dim=2,
        ).square(),
        dim=1
    ).sqrt()


@torch.jit.script
def tangent_norm_normalized(f: torch.Tensor) -> torch.Tensor:
    """Compute local similarity measure described in LeJeune et al., 2019.
        @param f batch-indexed tensor of (transposed) function of shape (N, *, K)
               with N batch size, * is an arbitrary number of input dimensions, K is the
               number of output dimensions.
    """
    nsamples = f.shape[0]
    ndirections = f.shape[1]
    batch_size = nsamples * ndirections
    f = batch_normalize(f, batch_size)
    f = f.view(nsamples, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,0].view(nsamples, 1, -1) - f[:,1:].view(nsamples, ndirections -1, -1)) / 0.1,
            p=2, dim=2
        ).square(),
        dim=1
    ).sqrt()


@torch.jit.script
def tangent_norm_euclidean(f: torch.Tensor) -> torch.Tensor:
    """Compute local similarity measure described in LeJeune et al., 2019.
        @param f batch-indexed tensor of (transposed) function of shape (N, *, Kl)
               with N batch size, * is an arbitrary number of input dimensions, Kl is the
               number of output dimensions.
    """
    nsamples = f.shape[0]
    nmcdirections = f.shape[1]
    ndirections = f.shape[2]
    f = f.reshape(nsamples, nmcdirections, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,:,0].view(nsamples, nmcdirections, 1, -1) - f[:,:,1:].view(nsamples, nmcdirections, ndirections -1, -1)) / 0.1,
            p=2, dim=3,
        ).square(),
        dim=2
    ).sqrt()


@torch.jit.script
def tangent_norm_euclidean_normalized(f: torch.Tensor) -> torch.Tensor:
    """Compute local similarity measure described in LeJeune et al., 2019.
        @param f batch-indexed tensor of (transposed) function of shape (N, *, K)
               with N batch size, * is an arbitrary number of input dimensions, K is the
               number of output dimensions.
    """
    nsamples = f.shape[0]
    nmcdirections = f.shape[1]
    ndirections = f.shape[2]
    batch_size = nsamples * nmcdirections * ndirections
    f = batch_normalize(f, batch_size)
    f = f.view(nsamples, nmcdirections, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,:,0].view(nsamples, nmcdirections, 1, -1) - f[:,:,1:].view(nsamples, nmcdirections, ndirections -1, -1)) / 0.1,
            p=2, dim=3
        ).square(),
        dim=2
    ).sqrt()


@torch.jit.script
def tangent_norm_geodesic(f: torch.Tensor) -> torch.Tensor:
    """ Integrate Jacobian differences along a piecewise linear path, 
        specified by a sequence of consecutive points
        
        @param f batch-indexed transposed function of shape (N, K, A, D, Kl), with N batch dimension,
               K MC sampling directions, A number of consecutive points for evaluating the line integral, 
               D function input dimension, Kl function output dimensions.
        
    """
    nsamples = f.shape[0]
    nmcdirections = f.shape[1]
    nanchors = f.shape[2]
    ndirections = f.shape[3]
    
    f = f.reshape(nsamples, nmcdirections, nanchors, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,:,:,0].view(nsamples, nmcdirections, nanchors, 1, -1) - f[:,:,:,1:].view(nsamples, nmcdirections, nanchors, ndirections -1, -1)) / 0.1,
            p=2, dim=4
        ).square(),
        dim=3
    ).sqrt()
    

@torch.jit.script
def tangent_norm_geodesic_normalized(f: torch.Tensor) -> torch.Tensor:
    """ Integrate Jacobian differences along a piecewise linear path, 
        specified by a sequence of consecutive points
        
        @param f batch-indexed transposed function of shape (N, K, A, D, C), with N batch dimension,
               K MC sampling directions, A number of consecutive points for evaluating the line integral, 
               D function input dimension, C function output dimensions.
        
    """
    nsamples = f.shape[0]
    nmcdirections = f.shape[1]
    nanchors = f.shape[2]
    ndirections = f.shape[3]
    f = batch_normalize(f, batch_size=nsamples * nmcdirections * nanchors * ndirections)
    f = f.reshape(nsamples, nmcdirections, nanchors, ndirections, -1)
    return torch.mean(
        torch.norm(
            (f[:,:,:,0].view(nsamples, nmcdirections, nanchors, 1, -1) - f[:,:,:,1:].view(nsamples, nmcdirections, nanchors, ndirections -1, -1)) / 0.1,
            p=2, dim=4
        ).square(),
        dim=3
    ).sqrt()

