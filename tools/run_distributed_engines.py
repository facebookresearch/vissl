# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
Supports SLURM as an option
"""

import sys
from typing import List, Any

from hydra.experimental import initialize_config_module, compose

from vissl.utils.distributed_training import is_submitit_available, launch_on_local_node, launch_on_slurm
from vissl.utils.hydra_config import is_hydra_available, convert_to_attrdict


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    if config.SLURM.ENABLED:
        assert is_submitit_available(), "Please 'pip install submitit' to schedule jobs on SLURM"
        launch_on_slurm(engine_name=args.engine_name, config=config)
    else:
        launch_on_local_node(node_id=args.node_id, engine_name=args.engine_name, config=config)


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
