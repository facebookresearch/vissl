# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features.
Supports SLURM as an option. Set config.SLURM.USE_SLURM=true to use slurm.
"""

import sys
from typing import List, Any

from hydra.experimental import initialize_config_module, compose
from vissl.hooks import default_hook_generator
from vissl.utils.distributed_launcher import (
    launch_distributed,
    launch_distributed_on_slurm,
)
from vissl.utils.hydra_config import is_hydra_available, convert_to_attrdict
from vissl.utils.slurm import is_submitit_available


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)

    if config.SLURM.USE_SLURM:
        assert (
            is_submitit_available()
        ), "Please 'pip install submitit' to schedule jobs on SLURM"
        launch_distributed_on_slurm(engine_name=args.engine_name, cfg=config)
    else:
        launch_distributed(
            cfg=config,
            node_id=args.node_id,
            engine_name=args.engine_name,
            hook_generator=default_hook_generator,
        )


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
