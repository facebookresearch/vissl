# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from vissl.models import build_model
from vissl.utils.checkpoint import (
    CheckpointFormatConverter,
    CheckpointLoader,
    CheckpointWriter,
)
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    init_distributed_on_file,
    with_temp_files,
)


class TestCheckpointConversion(unittest.TestCase):
    @staticmethod
    def _create_fsdp_model_config(with_fsdp: bool):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "+config/pretrain/swav/models=regnet16Gf",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.LOSS.swav_loss.epsilon=0.03",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.MODEL.FSDP_CONFIG.mixed_precision=False",
                "config.MODEL.FSDP_CONFIG.fp32_reduce_scatter=False",
                "config.MODEL.FSDP_CONFIG.compute_dtype=float32",
                "config.OPTIMIZER.construct_single_param_group_only=True",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
        return config

    @staticmethod
    def _worker(gpu_id: int, sync_file: str, world_size: int, with_heads: bool):
        torch.manual_seed(0)
        os.environ["RANK"] = str(gpu_id)
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        torch.backends.cudnn.deterministic = True

        config = TestCheckpointConversion._create_fsdp_model_config(with_fsdp=True)
        model = build_model(config.MODEL, config.OPTIMIZER).cuda(gpu_id)
        model = fsdp_wrapper(model, **config.MODEL.FSDP_CONFIG)
        optimizer = optim.SGD(model.parameters(), lr=1e-4)

        # Fake inputs
        num_iterations = 5
        batch_size = 3
        torch.manual_seed(gpu_id)
        fake_inputs = torch.randn(size=(num_iterations, batch_size, 3, 96, 96))
        fake_targets = torch.randn(size=(num_iterations, batch_size))

        # Fake training loop
        criterion = nn.MSELoss()
        for iteration in range(num_iterations):
            fake_input = fake_inputs[iteration].cuda(gpu_id)
            fake_target = fake_targets[iteration].cuda(gpu_id)
            output1, output2 = model(fake_input)[0]
            loss = criterion(output1.sum(axis=-1), fake_target) + criterion(
                output2.sum(axis=-1), fake_target
            )
            if gpu_id == 0:
                print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save a bunch of checkpoint, one by shard
        checkpoint_writer = CheckpointWriter(
            checkpoint_folder=".",
            is_final_train_phase=True,
            mode="iteration",
            mode_num=0,
            backend="disk",
        )
        content = {
            "classy_state_dict": {
                "base_model": {
                    "model": {
                        "trunk": model.trunk.local_state_dict(),
                        "heads": [head.local_state_dict() for head in model.heads],
                    },
                    "meta": {
                        "trunk": model.trunk.local_metadata_dict(),
                        "heads": [head.local_metadata_dict() for head in model.heads],
                    },
                }
            }
        }

        if not with_heads:
            del content["classy_state_dict"]["base_model"]["model"]["heads"]
            del content["classy_state_dict"]["base_model"]["meta"]["heads"]

        checkpoint_writer.save_sharded_checkpoint(
            content, shard_rank=gpu_id, world_size=world_size
        )
        dist.barrier()
        print(os.listdir("."))

        # Convert the checkpoint to consolidated and sliced checkpoints
        if gpu_id == 0:
            CheckpointFormatConverter.sharded_to_consolidated_checkpoint(
                "checkpoint.torch", "checkpoint_conso.torch"
            )
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                "checkpoint.torch", "checkpoint_sliced.torch"
            )
        dist.barrier()
        print(os.listdir("."))

        # Now create models initialized from the previous checkpoint and compare them
        fake_test_input = torch.randn(size=(1, 3, 96, 96)).cuda(gpu_id)

        shard_cp = CheckpointLoader.load_and_broadcast_init_weights(
            "checkpoint.torch", device=torch.device("cpu")
        )
        shard_model = build_model(config.MODEL, config.OPTIMIZER).cuda(gpu_id)
        shard_model = fsdp_wrapper(shard_model, **config.MODEL.FSDP_CONFIG)
        shard_model.init_model_from_weights_params_file(config, shard_cp)

        conso_cp = CheckpointLoader.load_and_broadcast_init_weights(
            "checkpoint_conso.torch", device=torch.device("cpu")
        )
        conso_model = build_model(config.MODEL, config.OPTIMIZER).cuda(gpu_id)
        conso_model = fsdp_wrapper(conso_model, **config.MODEL.FSDP_CONFIG)
        conso_model.init_model_from_weights_params_file(config, conso_cp)

        slice_cp = CheckpointLoader.load_and_broadcast_init_weights(
            "checkpoint_sliced.torch", device=torch.device("cpu")
        )
        slice_model = build_model(config.MODEL, config.OPTIMIZER).cuda(gpu_id)
        slice_model = fsdp_wrapper(slice_model, **config.MODEL.FSDP_CONFIG)
        slice_model.init_model_from_weights_params_file(config, slice_cp)

        # Verifying that the models are equivalent
        if gpu_id == 0:
            slice_state_dict = slice_model.local_state_dict()
            conso_state_dict = conso_model.local_state_dict()
            assert set(slice_state_dict.keys()) == set(conso_state_dict.keys())
            for k in slice_state_dict.keys():
                slice_val = slice_state_dict[k]
                conso_val = conso_state_dict[k]
                assert torch.allclose(
                    slice_val, conso_val
                ), f"Difference for key {k}: {slice_val} VS {conso_val}"
        dist.barrier()

        with torch.no_grad():
            ref_out = model.trunk(fake_test_input)[0]
            shard_out = shard_model.trunk(fake_test_input)[0]
            conso_out = conso_model.trunk(fake_test_input)[0]
            slice_out = slice_model.trunk(fake_test_input)[0]
            assert torch.allclose(
                ref_out, shard_out
            ), f"{ref_out.sum()} vs {shard_out.sum()}"
            assert torch.allclose(
                ref_out, conso_out
            ), f"{ref_out.sum()} vs {conso_out.sum()}"
            assert torch.allclose(
                ref_out, slice_out
            ), f"{ref_out.sum()} vs {slice_out.sum()}"

    @gpu_test(gpu_count=2)
    def test_checkpoint_consolidation(self) -> None:
        with in_temporary_directory():
            for with_heads in [True, False]:
                with with_temp_files(count=1) as sync_file:
                    world_size = 2
                    mp.spawn(
                        self._worker,
                        (sync_file, world_size, with_heads),
                        nprocs=world_size,
                    )
