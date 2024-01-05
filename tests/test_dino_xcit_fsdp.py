# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from vissl.models import build_model
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    init_distributed_on_file,
    run_integration_test,
    with_temp_files,
)


class TestDINO_XCIT(unittest.TestCase):
    @staticmethod
    def _create_dino_pretraining_config(
        with_fsdp: bool,
        with_mixed_precision: bool,
        gpu_count: int = 2,
        num_epochs: int = 4,
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino_xcit",
                "config.SEED_VALUE=0",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                f"config.OPTIMIZER.num_epochs={num_epochs}",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
                # Options to override to get FSDP
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                f"config.MODEL.AMP_PARAMS.USE_AMP={with_mixed_precision}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "xcit_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "dino_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.flatten_parameters = False
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.TRUNK.XCIT.CHECKPOINT_BLOCK = True
        return config

    def run_config(self, config):
        with in_temporary_directory():
            result = run_integration_test(config)
            return result.get_losses()

    @gpu_test(gpu_count=2)
    def test_init_vit_init(self) -> None:
        """
        Check that the initialisation of the ViT with FSDP leads
        to the same results than the initialisation with DDP
        """
        with in_temporary_directory():
            with with_temp_files(count=1) as sync_file:
                world_size = 2
                mp.spawn(
                    self._worker_test_init_vit_init,
                    (sync_file, world_size),
                    nprocs=world_size,
                )

    @classmethod
    def _worker_test_init_vit_init(cls, gpu_id: int, sync_file: str, world_size: int):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        ddp_config = cls._create_dino_pretraining_config(
            with_fsdp=False, with_mixed_precision=False, gpu_count=world_size
        )
        fsdp_config = cls._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=False, gpu_count=world_size
        )

        # For faster tests, try a smaller ViT
        ddp_config.MODEL.TRUNK.XCIT.NUM_LAYERS = 2
        fsdp_config.MODEL.TRUNK.XCIT.NUM_LAYERS = 2

        # For deterministic computing
        ddp_config.MODEL.TRUNK.XCIT.DROP_PATH_RATE = 0.0
        fsdp_config.MODEL.TRUNK.XCIT.DROP_PATH_RATE = 0.0

        # Create fake inputs
        torch.random.manual_seed(gpu_id)
        x = torch.randn(size=(2, 3, 224, 224)).cuda()

        # Create both FSDP and DDP models
        ddp_model = build_model(ddp_config["MODEL"], ddp_config["OPTIMIZER"]).cuda()
        ddp_model = DistributedDataParallel(ddp_model, device_ids=[gpu_id])
        fsdp_model = build_model(fsdp_config["MODEL"], fsdp_config["OPTIMIZER"]).cuda()
        fsdp_model = fsdp_wrapper(fsdp_model, **fsdp_config["MODEL"]["FSDP_CONFIG"])
        if gpu_id == 0:
            print(fsdp_model)

        # Check that the weights are equal on both models
        with fsdp_model.summon_full_params():
            ddp_params = {n: p.sum().item() for n, p in ddp_model.named_parameters()}
            for n, p in fsdp_model.named_parameters():
                p_ref = ddp_params[f"module.{n}"]
                assert p_ref == p.sum().item()

        # Check that the local_state_dict / state_dict are valid
        fsdp_model_2 = build_model(
            fsdp_config["MODEL"], fsdp_config["OPTIMIZER"]
        ).cuda()
        fsdp_model_2 = fsdp_wrapper(fsdp_model_2, **fsdp_config["MODEL"]["FSDP_CONFIG"])
        fsdp_model_2.load_local_state_dict(fsdp_model.local_state_dict())
        fsdp_model_2.load_state_dict(fsdp_model.state_dict())

        # Compare that DDP and FSDP models give the same results
        with torch.no_grad(), torch.cuda.amp.autocast():
            ddp_out = ddp_model(x)
            fsdp_out = fsdp_model(x)
            if gpu_id == 0:
                print(ddp_out[0][0])
                print(fsdp_out[0][0])
            assert torch.allclose(ddp_out[0][0], fsdp_out[0][0])
