import sys
from typing import List, Any

from hydra.experimental import initialize_config_module, compose

from vissl.utils.distributed_launcher import schedule_on_slurm
from vissl.utils.hydra_config import is_hydra_available, convert_to_attrdict


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)

    args, config = convert_to_attrdict(cfg)
    config.DATA.NUM_DATALOADER_WORKERS = 8
    schedule_on_slurm(
        engine_name=args.engine_name,
        config=config,
        job_name=args.name,
        job_comment=args.comment,
        partition=args.partition,
        time_hours=args.time_hours,
        constraint=args.constraint,
        mem_gb=args.mem_gb,
        log_folder=args.log_folder)


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_on_slurm.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
