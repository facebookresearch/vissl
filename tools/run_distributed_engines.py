# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features.
Supports SLURM as an option. Set config.SLURM.USE_SLURM=true to use slurm.
"""
import os
import sys
from typing import Any, List

from vissl.utils.distributed_launcher import (
    launch_distributed,
    launch_distributed_on_slurm,
)
from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    SweepHydraOverrides,
)
from vissl.utils.io import makedir
from vissl.utils.slurm import is_submitit_available


def hydra_main(overrides: List[Any]):
    ######################################################################################
    # DO NOT MOVE THIS IMPORT TO TOP LEVEL: submitit processes will not be initialized
    # correctly (MKL_THREADING_LAYER will be set to INTEL instead of GNU)
    ######################################################################################
    from vissl.hooks import default_hook_generator

    ######################################################################################

    print(f"####### overrides: {overrides}")
    cfg = compose_hydra_configuration(overrides)
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


def hydra_multirun_main(overrides: List[Any]):
    """
    Utility function which allows to run multiple jobs with different
    hyper-parameters, using the syntax of Hydra sweep overrides:

    ```
    python tools/run_distributed_engines.py
        config=path/to/configuration
        config.OPTIMIZER.weight_decay=0.001,0.0001
        config.OPTIMIZER.num_epochs=50,100,200
    ```

    This command will run 6 jobs, with all the cross combinations of
    weight_decay and num_epochs, setting the checkpoint directory
    and the running directory to sub-folders of the current directory
    """
    overrides, sweeps = SweepHydraOverrides.from_overrides(overrides)
    if not sweeps:
        hydra_main(overrides)
    else:
        multirun_dir = os.getcwd()
        for sweep_id, sweep_overrides in enumerate(sweeps):
            run_dir = os.path.join(multirun_dir, f"sweep_{sweep_id}")
            makedir(run_dir)
            os.chdir(run_dir)
            checkpoint_path_overrides = [f"config.CHECKPOINT.DIR={run_dir}"]
            hydra_main(overrides + sweep_overrides + checkpoint_path_overrides)


def main() -> None:
    global overrides
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    hydra_multirun_main(overrides=overrides)


if __name__ == "__main__":
    main()  # pragma: no cover
