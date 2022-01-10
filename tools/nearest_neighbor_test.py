# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import Namespace
from typing import Any, List

from vissl.config import AttrDict
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    print_cfg,
)
from vissl.utils.knn_utils import run_knn_at_all_layers
from vissl.utils.logger import setup_logging, shutdown_logging


def main(args: Namespace, config: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the coniguration used
    print_cfg(config)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # Extract the features if no path to the extract features is provided
    if not config.NEAREST_NEIGHBOR.FEATURES.PATH:
        launch_distributed(
            config,
            args.node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )
        config.NEAREST_NEIGHBOR.FEATURES.PATH = get_checkpoint_folder(config)

    # Run KNN at all the extract features
    run_knn_at_all_layers(config)

    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)
