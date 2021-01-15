# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import sys
from typing import Any, List

from hydra.experimental import compose, initialize_config_module

from vissl.hooks import default_hook_generator
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available
from vissl.utils.logger import setup_logging, shutdown_logging


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    setup_logging(__name__)
    args, config = convert_to_attrdict(cfg)
    launch_distributed(
        config,
        node_id=args.node_id,
        engine_name=args.engine_name,
        hook_generator=default_hook_generator,
    )
    # close the logging streams including the filehandlers
    shutdown_logging()


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
