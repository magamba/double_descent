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

import abc
from enum import Enum
import logging
import functools
import torch.nn as nn
import torch
import numpy as np

from core.metric_helpers import (
    get_jacobian_fn,
    get_tangent_norm_fn,
    accuracy,
    get_model_forward,
)

logger = logging.getLogger(__name__)

class Metric(nn.Module, abc.ABC):
    """Base Metric class.
    
       Each metric is implemented as a torch.nn.Module and
       the metric computation is done within the forward method.
       
       When enabled, Monte Carlo integration is performed within each metric.
       
       Irrespective of the sampler used, a metric should always 
       return batched results of shape (N,), where N is the number
       of base data points.
       
    """
    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(Metric, self).__init__()
        self.device = torch.device("cuda:0") if cmd_args.device != "cpu" and torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device, non_blocking=True)
        self.model = self.model.eval()
        self._tangent_metric = is_tangent_metric(self)
        
        # batch shape attributes
        assert len(data_shape) > 3, "Error: expecting batches of shape (*, C, H, W), got: {}".format(tuple(data_shape))
        self._data_shape = data_shape
        self._batch_dims = data_shape[:-4] if self._tangent_metric else data_shape[:-3] 
        self._batch_size = np.prod(self._batch_dims)
        self._nbatch_dims = len(self._batch_dims)
        self._flatten_batch_dims = (-1,) + tuple(data_shape[self._nbatch_dims+1:]) if self._tangent_metric else (-1,) + tuple(data_shape[self._nbatch_dims:])
        self._normalize_dims = self._batch_dims
        self._tangent_batch_size = self._data_shape[-4] if self._tangent_metric else None
        
        self._requires_targets = False
        self._disable_geodesic_integration = cmd_args.disable_geodesic_integration
        self._cumulative_geodesic_integration = cmd_args.cumulative_geodesic_integration
        self.integrate_fn = self._set_mc_integration_fn()
        self._init_args(cmd_args, **kwargs)

    def _init_args(self, cmd_args, **kwargs):
        """Used to pass command-line arguments to each metric.
        """
        pass
    
    def _set_mc_integration_fn(self):
        """Set up function used for Monte Carlo integration
           of the metric.
           
           This essentially ties a metric instantiation to a sampling
           strategy:
            - for BaseSamplers, no MC integration is performed
            - for EuclideanSamplers, MC integration is performed in the second
              batch dimension K, with input batch shape (N, K, *).
            - for GeodesicSamplers, MC integration is performed in the third
              batch dimension A, with input batch shape (N, K, A, *).
        """
        if self._nbatch_dims < 3 and self._disable_geodesic_integration:
            raise ValueError("Error: geodesic integration can only be enabled/disabled for geodesic sampling strategies.")
        
        if self._nbatch_dims == 1:
            # base sampler
            return None
        
        if self._nbatch_dims == 2:
            # Euclidean MC sampler
            from core.metric_helpers import integrate
            def mc_integrate(f, x):
                """Integrate f(x) given a tensor of samples @f
                """
                return integrate(
                    f.view(
                        self._normalize_dims
                    )
                )
            
        elif self._nbatch_dims == 3:
            # Geodesic MC sampler
            from core.metric_helpers import integrate_line as integrate, path_distances, integrate_line_cumulative as integrate_cumulative
            nsamples, ndirections, nanchors = tuple(self._batch_dims)
            
            if self._tangent_metric:
                def strip_tangent_batch_dim(x):
                    x = x[...,0,:,:,:]
                    return x.view(self._batch_dims + tuple(x.shape[self._nbatch_dims:]))
            else:
                def strip_tangent_batch_dim(x):
                    return x.view(self._batch_dims + tuple(x.shape[self._nbatch_dims:]))
            
            if self._disable_geodesic_integration:
                def mc_integrate(f, x):
                    """ Rather than computing the line integral of f(x) over x = x(t),
                        return the values of f(x(t)), and ||x(t+1) - x(t)||.
                    """
                    x = strip_tangent_batch_dim(x)
                    return (
                        f.view(self._normalize_dims),
                        path_distances(
                            x, nsamples=nsamples, ndirections=ndirections, nanchors=nanchors
                        )
                    )
            elif self._cumulative_geodesic_integration:
                def mc_integrate(f, x):
                    """ Rather than computing the line integral of f(x) over x = x(t),
                        return the values of f(x(t)), and ||x(t+1) - x(t)||.
                    """
                    x = strip_tangent_batch_dim(x)
                    return integrate_cumulative(
                        f.view(self._normalize_dims),
                        path_distances(
                            x, nsamples=nsamples, ndirections=ndirections, nanchors=nanchors
                        )
                    )
            else:
                def mc_integrate(f, x):
                    """Line integral of f(x) over a curve x = x(t),
                       given corresponding samples of @f and @x
                    """
                    x = strip_tangent_batch_dim(x)
                    return integrate(
                        f.view(
                            self._normalize_dims
                        ),
                        path_distances(
                            x, nsamples=nsamples, ndirections=ndirections, nanchors=nanchors
                        )
                    )
        else:
            raise ValueError("Unsupported sampler batch shape: {}.".format(self._batch_dims))
        
        return mc_integrate

    @property
    def requires_targets(self):
        """ Return True if computing the metric requires access to targets.
        """
        return self._requires_targets
        
    @property
    @abc.abstractmethod
    def name(self):
        """Return the metric name for logging.
        """
        
    @abc.abstractmethod
    def forward_impl_(self, x, target=None):
        """Compute metric here
        
           A metric should always return batched results of shape
           (N, *), where N is the batch size, and * is any arbitrary
           number of output dimensions.
        """
        
    def forward(self, x, target=None):
        """ Compute metric and integrate
        """
        result_ = self.forward_impl_(x, target)
        if self.integrate_fn is not None:
            # integrated values might be tuples, 
            # while result_ is always a singleton
            result = self.integrate_fn(
                result_.type(torch.float32), 
                x
            )
        else:
            result = result_
        return result

    def __getstate__(self):
        """Delete model before serialization.
        """
        state = self.__dict__.copy()
        for key in ["model"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state

    def __repr__(self):
        """Return string representation of metric and model
        """
        return "{}:\n{}".format(self.name(), self.model)


def mask_last_anchor(path_dist_fn):
    """ When line-integrating a tangent metric, mask out the last
        anchor point of path_dist_fn, since it is invalid
        
        Note: this assumes that line integrals are approximated using
        the trapezoidal rule.
        
        Don't use this decorator if integration is performed using 
        left-Riemann integrals
    """
    @functools.wraps(path_dist_fn)
    def wrap_distance_function(*args, **kwargs):
        return path_dist_fn(*args, **kwargs)[:,:,:-1]
    return wrap_distance_function

        
class CrossEntropy(Metric):
    """Cross-entropy loss"""

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(CrossEntropy, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._requires_targets = True
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
    
    def name(self):
        return Metrics.CROSSENTROPY.value

    def forward_impl_(self, x, target):
        with torch.no_grad():
            out = self.model(
                x.view(self._flatten_batch_dims)
            )
            repeat_factor = out.shape[0] // target.shape[0]
            loss = self.criterion(out, target.repeat_interleave(repeat_factor)).view(self._batch_dims)
        return loss

        
class Accuracy(Metric):
    """top-1 0/1 loss"""

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(Accuracy, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._requires_targets = True

    def name(self):
        return Metrics.ACCURACY.value
        
    def forward_impl_(self, x, target):
        with torch.no_grad():
            out = self.model(
                x.view(self._flatten_batch_dims)
            )
            acc = accuracy(out, target).view(self._batch_dims).type(torch.int)
        return acc


class TangentJacobianNorm(Metric):
    """ Computes Jacobian norm
    """

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(TangentJacobianNorm, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._model_forward = get_model_forward(model, cmd_args.normalization)
        self._normalize_jacobian = cmd_args.normalization == "jacobian"
        self._requires_targets = cmd_args.target_logit_only or cmd_args.normalization == "crossentropy"
        self._tangent_norm_fn = get_tangent_norm_fn(data_shape, self._normalize_jacobian)
        assert self._tangent_batch_size > 2, "Error: tangent metrics require at least 2 points to be computed."
        if self._nbatch_dims == 2:
            self._normalize_dims = (self._batch_dims[0], self._batch_dims[1], -1)

    def name(self):
        return Metrics.TANGENT_JACOBIAN_NORM.value

    def forward_impl_(self, x, target=None):
        """ Compute the norm of the tangent Jacobian
        """
        
        if self._requires_targets:
            output = self._model_forward(
                x.view(self._flatten_batch_dims), target.to(self.device, non_blocking=True)
            )
        else:
            output = self._model_forward(x.view(self._flatten_batch_dims))

        jacobian_norm = self._tangent_norm_fn(
            output.view(self._batch_dims + (self._tangent_batch_size, -1,))
        )

        return jacobian_norm


class JacobianMetric(Metric):
    """Base class for evaluating Jacobian of a model w.r.t. to its input
    """
        
    def _init_args(self, cmd_args, **kwargs):
        super(JacobianMetric, self)._init_args(cmd_args, **kwargs)
        nclasses = kwargs.pop("nclasses", 0)
        assert nclasses > 0, "Missing required argument for {} metric: nclasses".format(self.name())
        self._target_logit_only = cmd_args.target_logit_only
        self._normalize_jacobian = cmd_args.normalization == "jacobian"
        self._requires_targets = cmd_args.target_logit_only or cmd_args.normalization == "crossentropy"
        effective_bs = self._batch_size * self._tangent_batch_size if self._tangent_metric else self._batch_size
        self._jacobian_fn = get_jacobian_fn(
            self.model, effective_bs, nclasses, cmd_args.bigmem, target_logit_only=self._target_logit_only, normalization=cmd_args.normalization,
        )


class JacobianNorm(JacobianMetric):
    """ Compute the Jacobian norm, optionally smoothed via MC integration
    """
    
    def name(self):
        return Metrics.JACOBIAN_NORM.value
    
    def forward_impl_(self, x, target=None):
        """ Compute the Jacobian norm of self.model w.r.t. @x
            optionally restricting computation to the specified @target output
            dimensions.
        """        
        if self._requires_targets:
            jacobian = self._jacobian_fn(
                x.view(self._flatten_batch_dims), target.to(self.device, non_blocking=True)
            )
        else:
            jacobian = self._jacobian_fn(x.view(self._flatten_batch_dims))
        
        jacobian_norm = torch.norm(
            jacobian.view(self._batch_dims + (-1,)),
            p=2,
            dim=self._nbatch_dims,
        )
        
        return jacobian_norm


class TangentHessianNorm(JacobianMetric):
    """Compute the tangent Hessian norm
    """
    
    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(TangentHessianNorm, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._tangent_norm_fn = get_tangent_norm_fn(data_shape, self._normalize_jacobian)
        assert self._tangent_batch_size > 1, "Error: tangent metrics require at least 2 points to be computed."
        if self._nbatch_dims == 2:
            self._normalize_dims = (self._batch_dims[0], self._batch_dims[1], -1)
   
    def name(self):
        return Metrics.TANGENT_HESSIAN_NORM.value
    
    def forward_impl_(self, x, target=None):
        """ Compute the norm of the tangent Hessian
        """
            
        if self._requires_targets:
            jacobian = self._jacobian_fn(
                x.view(self._flatten_batch_dims), target.to(self.device, non_blocking=True)
            )
        else:
            jacobian = self._jacobian_fn(x.view(self._flatten_batch_dims))
       
        hessian_norm = self._tangent_norm_fn(
            jacobian.view(self._batch_dims + (self._tangent_batch_size, -1,))
        )
        
        return hessian_norm


def is_tangent_metric(obj):
    return isinstance(obj, (TangentJacobianNorm, TangentHessianNorm))


class Metrics(Enum):
    TANGENT_JACOBIAN_NORM = "tangent_jacobian"
    TANGENT_HESSIAN_NORM = "tangent_hessian"
    JACOBIAN_NORM = "jacobian"
    ACCURACY = "accuracy"
    CROSSENTROPY = "crossentropy"


METRICS = {
    Metrics.TANGENT_JACOBIAN_NORM.value: TangentJacobianNorm,
    Metrics.TANGENT_HESSIAN_NORM.value: TangentHessianNorm,
    Metrics.JACOBIAN_NORM.value: JacobianNorm,
    Metrics.ACCURACY.value: Accuracy,
    Metrics.CROSSENTROPY.value: CrossEntropy,
}


def create_metric(metric_name, model, data_shape, cmd_args=None, **kwargs):
    metric_factory = METRICS[metric_name]
    return metric_factory(model, data_shape, cmd_args=cmd_args, **kwargs)
