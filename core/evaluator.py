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

import logging
import torch

""" Evaluators take a metric, a sampler, and evaluate the metric on each 
    batch yielded by the sampler.
    
    Evaluators are independent from specific data streams,
    but metrics can assume a particular batch shape.
    
    However, a metric should always return batched results of shape (N, *),
    where N is the number of base data points.
    
    When defining new evaluators, a user should take care of matching a metric
    with a supported sampler.
    
    See core/metrics.py for available metrics, and core/sampling.py for
    samplers.
"""

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, metric, sampler, cmd_args, **kwargs):
        self.metric = metric
        self.sampler = sampler
        self._batch_size = sampler.data_shape[0]
        self._requires_targets = metric.requires_targets
        self.results = {
            "sample_ids" : [],
            "targets" : [],
            "ground_truths" : [],
            "stats" : [],
        }
        self._init_args(cmd_args, **kwargs)

    def _init_args(self, cmd_args, **kwargs):
        """Used to pass command-line arguments to the constructor.
        """
        self.device = torch.device("cuda:0") if cmd_args.device != "cpu" and torch.cuda.is_available() else torch.device("cpu")
        self._disable_geodesic_integration = cmd_args.disable_geodesic_integration
        self._restored_targets = cmd_args.restored_targets

    def __getstate__(self):
        """Delete sampler and metric before serialization.
        """
        state = self.__dict__.copy()
        for key in ["sampler", "metric"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state

    def __repr__(self):
        """ Return string representation of results
        """
        str_dict = {
            "metric": self.metric.name(),
            "sampler": self.sampler.name(),
            "results": self.results
        }
        return str(str_dict)

    def run(self, **kwargs) -> int:
        """Evaluate @self.metric on @self.sampler
        """
        sample_ids = []
        targets = []
        ground_truths = []
        stats = []
        path_distances = []
        uid = self.sampler.uid # unique id for data parallelism

        logger.info("Computing metric {}".format(self.metric.name()))

        for (x, target, sample_idx, ground_truth) in self.sampler:
            x = x.to(self.device, non_blocking=True)
            
            if self._requires_targets:
                if self._restored_targets:
                    labels = ground_truth
                else:
                    labels = target
                
                labels = labels.to(self.device, non_blocking=True)
                stat_ = self.metric(x, target=labels).view(
                    self._batch_size, -1
                )
                labels = labels.to("cpu", non_blocking=False)
            else:
                stat_ = self.metric(x).view(
                    self._batch_size, -1
                )
            
            if self._disable_geodesic_integration:
                stat, path_lengths = stat_
                path_lengths = path_lengths.to("cpu", non_blocking=False)
                path_distances += path_lengths.tolist()
            else:
                stat = stat_
            
            stat = stat.to("cpu", non_blocking=False)
            sample_ids += sample_idx.tolist()
            ground_truths += ground_truth.tolist()
            stats += stat.tolist()
            targets += target.tolist()
        
        self.results["sample_ids"] = sample_ids
        self.results["targets"] = targets
        self.results["ground_truths"] = ground_truths
        self.results["stats"] = stats
        
        if len(path_distances) > 0:
            self.results["path_distances"] = path_distances
        
        logger.info("Done with metric {}".format(self.metric.name()))
        
        return uid


def load_evaluator(metric, sampler, cmd_args, **kwargs):
    return Evaluator(metric, sampler, cmd_args, **kwargs)
