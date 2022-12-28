# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from vissl.models import build_model
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker
from vissl.utils.profiler import create_runtime_profiler
from vissl.utils.test_utils import (
    in_temporary_directory,
    init_distributed_on_file,
    with_temp_files,
    with_timing,
)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Output path for artifacts")
    parser.add_argument(
        "-t",
        "--trace",
        action="store_const",
        const=True,
        default=False,
        help="trace memory usage",
    )
    parser.add_argument(
        "-p",
        "--profile",
        action="store_const",
        const=True,
        default=False,
        help="enable profiling",
    )
    parser.add_argument(
        "-a",
        "--amp",
        type=str,
        default="",
        help="Automatic Mixed Precision: either nothing, 'O1' or 'O2'",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Batch size used to bench the model",
    )
    return parser


@dataclass
class BenchConfig:
    output_path: str
    track_memory: bool
    enable_profiler: bool
    batch_size: int
    amp_flag: str


class BenchVitFSDP:
    @staticmethod
    def _create_dino_pretraining_config(
        with_fsdp: bool, with_mixed_precision: bool, gpu_count: int = 2
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/performance/dino_vit",
                "config.SEED_VALUE=0",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.OPTIMIZER.num_epochs=4",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
                # Options to override to get FSDP
                "config.MODEL.TRUNK.NAME=vision_transformer",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                f"config.MODEL.AMP_PARAMS.USE_AMP={with_mixed_precision}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "vision_transformer_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "dino_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
        return config

    @staticmethod
    def _print_memory_used():
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")

    @classmethod
    def _run_fsdp_worker(
        cls,
        gpu_id: int,
        sync_file: str,
        config,
        world_size: int,
        bench_config: BenchConfig,
    ):
        with_memory_tracking = bench_config.track_memory
        with_profiling = bench_config.enable_profiler

        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        if with_memory_tracking and gpu_id == 0:
            config.MODEL.FSDP_CONFIG._TRACK_COMMUNICATIONS = True

        model = build_model(config.MODEL, config.OPTIMIZER).cuda(gpu_id)
        model = fsdp_wrapper(model, **config.MODEL.FSDP_CONFIG)

        criterion = nn.CrossEntropyLoss()
        if gpu_id == 0:
            print(model)

        batch_size = bench_config.batch_size
        device = torch.device(gpu_id)
        batch = torch.zeros(size=(batch_size, 3, 224, 224), device=device)
        targets = torch.tensor([0] * batch_size, dtype=torch.int64, device=device)

        autocast_context = (
            torch.cuda.amp.autocast()
            if config.MODEL.AMP_PARAMS.USE_AMP
            else contextlib.suppress()
        )

        def forward_backward():
            with autocast_context:
                outputs = model(batch)
                loss = criterion(outputs[0][0], targets)
                for output in outputs[1:]:
                    loss += criterion(output, targets)
            loss.backward()  # Not recommended to put in auto-cast

        # First forward-backward passes to warm up the system
        for _ in range(2):
            forward_backward()

        if with_memory_tracking and gpu_id == 0:
            memory_tracker = LayerwiseMemoryTracker()
            memory_tracker.monitor(model)
        else:
            memory_tracker = None

        if with_profiling and gpu_id == 0:
            runtime_profiler = create_runtime_profiler(
                enabled=True,
                use_cpu=False,
                use_cuda=True,
                wait=0,
                warmup=0,
                active=1,
                legacy_profiler=False,
            )
        else:
            runtime_profiler = None

        with with_timing(f"forward/backward (rank {gpu_id})"):
            if runtime_profiler:
                runtime_profiler.__enter__()
            forward_backward()
            if runtime_profiler:
                runtime_profiler.__exit__(None, None, None)

        if gpu_id == 0:
            cls._print_memory_used()

        # Dump content of the profiler
        if runtime_profiler:
            runtime_profiler.dump(bench_config.output_path, rank=0)

        # Print a summary of the tracker
        if memory_tracker:
            memory_tracker.stop()
            image = memory_tracker.show_plots(capture=True)
            image.save(f"{bench_config.output_path}/memory_tracking.jpg")
            memory_tracker.save_traces(
                f"{bench_config.output_path}/memory_tracking.json"
            )

    def bench_fsdp_vit(self, bench_config: BenchConfig):
        world_size = torch.cuda.device_count()
        # TODO - add config with_mixed_precision
        # TODO - add different configs
        config = self._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=False, gpu_count=world_size
        )
        with in_temporary_directory():
            with with_temp_files(count=1) as sync_file:
                args = (sync_file, config, world_size, bench_config)
                mp.spawn(self._run_fsdp_worker, args, nprocs=world_size)


if __name__ == "__main__":
    """
    Run this benchmark on a 8 GPU nodes for good benchmark results
    """
    args = argument_parser().parse_args()
    bench_config = BenchConfig(
        output_path=args.output,
        track_memory=args.trace,
        enable_profiler=args.profile,
        batch_size=args.batch_size,
        amp_flag=args.amp,
    )
    BenchVitFSDP().bench_fsdp_vit(bench_config)
