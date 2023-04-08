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

""" Monte Carlo sampling strategies
"""

import logging
from functools import partial
from core.sampling import create_sampler

logger = logging.getLogger(__name__)

def load_sampler(cmd_args):
    """ Initialize MC sampling strategy
    """
    logger.debug("Initializing strategy: {}".format(cmd_args.gen_strategy))
    try:
        strategy = GEN_STRATEGIES[cmd_args.gen_strategy]
    except KeyError:
        raise ValueError("Unrecognized strategy: {}".format(cmd_args.gen_strategy))
        
    return strategy(cmd_args=cmd_args)


GEN_STRATEGIES = {
    "none" : partial(create_sampler, "none", None, None), # disable MC sampling
    "1px-shifts" : partial(create_sampler, "euclidean", "shifts", [1]),
    "4px-shifts" : partial(create_sampler, "euclidean", "shifts", [4]),
    "svd-5" : partial(create_sampler, "euclidean", "svd", [5]),
    "svd-10" : partial(create_sampler, "euclidean", "svd", [10]),
    "svd-15" : partial(create_sampler, "euclidean", "svd", [15]),
    "svd-20" : partial(create_sampler, "euclidean", "svd", [20]),
    "svd-22" : partial(create_sampler, "euclidean", "svd", [22]),
    "svd-25" : partial(create_sampler, "euclidean", "svd", [25]),
    "shifts-path" : partial(create_sampler, "geodesic", "shifts", [1, 2, 3, 4]),
    "svd-path" : partial(create_sampler, "geodesic", "svd", [5, 10, 15, 20, 22, 25]),
}

