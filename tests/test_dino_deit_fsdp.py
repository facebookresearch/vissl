# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from vissl.models import build_model
from vissl.models.heads.dino_head import DINOHead, DINOHeadFSDP
from vissl.utils.checkpoint import CheckpointFormatConverter, DINOCheckpointUtils
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.misc import set_torch_seed
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    init_distributed_on_file,
    run_integration_test,
    with_temp_files,
)


class TestDINO(unittest.TestCase):
    """
    Unit and integration tests to check that the behavior of DINO is correct
    under FSDP.
    """

    @staticmethod
    def _create_dino_pretraining_config(
        with_fsdp: bool,
        with_mixed_precision: bool,
        with_multicrop: bool = False,
        with_mlp_checkpoints: bool = False,
        with_block_checkpoints: bool = False,
        gpu_count: int = 2,
        num_epochs: int = 2,
    ):
        main_config = "config=test/integration_test/quick_dino"
        if with_multicrop:
            main_config = "config=test/integration_test/quick_dino_multicrop"

        cfg = compose_hydra_configuration(
            [
                main_config,
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
                # Activation checkpointing (memory optimisation)
                f"config.MODEL.TRUNK.VISION_TRANSFORMERS.CHECKPOINT_MLP={with_mlp_checkpoints}",
                f"config.MODEL.TRUNK.VISION_TRANSFORMERS.CHECKPOINT_BLOCK={with_block_checkpoints}",
                # Options to override to get FSDP
                "config.MODEL.TRUNK.NAME=vision_transformer",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                f"config.MODEL.AMP_PARAMS.USE_AMP={with_mixed_precision}",
                "config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "vision_transformer_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "dino_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.flatten_parameters = False
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
        return config

    @staticmethod
    def _create_dino_linear_eval_config(
        checkpoint_path: str, with_fsdp: bool, gpu_count: int = 2
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_in1k_linear.yaml",
                "+config/benchmark/linear_image_classification/imagenet1k/models=dino_deit_s16",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
                # Datasets
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TEST.USE_DEBUGGING_SAMPLER=True",
                # To get the logs reliably
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.OPTIMIZER.num_epochs=2",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "vision_transformer_fsdp"
            num_heads = len(config["MODEL"]["HEAD"]["PARAMS"])
            for i in range(num_heads):
                head_name = config["MODEL"]["HEAD"]["PARAMS"][i][0]
                config["MODEL"]["HEAD"]["PARAMS"][i][0] = f"{head_name}_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.flatten_parameters = False
            config.MODEL.FSDP_CONFIG.mixed_precision = False
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = False
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
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
            with_fsdp=False, with_mixed_precision=True, gpu_count=world_size
        )
        fsdp_config = cls._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=True, gpu_count=world_size
        )

        # For faster tests, try a smaller ViT
        ddp_config.MODEL.TRUNK.VISION_TRANSFORMERS.NUM_LAYERS = 2
        fsdp_config.MODEL.TRUNK.VISION_TRANSFORMERS.NUM_LAYERS = 2

        # Create fake inputs
        torch.random.manual_seed(gpu_id)
        x = torch.randn(size=(1, 3, 224, 224)).cuda()

        # Create both FSDP and DDP models
        ddp_model = build_model(ddp_config["MODEL"], ddp_config["OPTIMIZER"]).cuda()
        ddp_model = DistributedDataParallel(ddp_model, device_ids=[gpu_id])
        fsdp_model = build_model(fsdp_config["MODEL"], fsdp_config["OPTIMIZER"]).cuda()
        fsdp_model = fsdp_wrapper(fsdp_model, **fsdp_config["MODEL"]["FSDP_CONFIG"])
        if gpu_id == 0:
            print(fsdp_model)

        # Check that the local_state_dict / state_dict are valid
        fsdp_model_2 = build_model(
            fsdp_config["MODEL"], fsdp_config["OPTIMIZER"]
        ).cuda()
        fsdp_model_2 = fsdp_wrapper(fsdp_model_2, **fsdp_config["MODEL"]["FSDP_CONFIG"])
        if gpu_id == 0:
            fsdp_model.load_local_state_dict(fsdp_model_2.local_state_dict())
        fsdp_model.load_state_dict(fsdp_model_2.state_dict())

        # Compare that DDP and FSDP models give the same results
        with torch.no_grad(), torch.cuda.amp.autocast():
            ddp_out = ddp_model(x)
            fsdp_out = fsdp_model(x)
            assert torch.allclose(ddp_out[0][0], fsdp_out[0][0])

    @gpu_test(gpu_count=2)
    def test_dino_head(self) -> None:
        """
        Check that the FSDP SwAV head reacts identically as the DDP one
        """
        with in_temporary_directory():
            with with_temp_files(count=1) as sync_file:
                world_size = 2
                mp.spawn(
                    self._dino_head_worker, (sync_file, world_size), nprocs=world_size
                )

    @classmethod
    def _dino_head_worker(cls, gpu_id: int, sync_file: str, world_size: int):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        ddp_config = cls._create_dino_pretraining_config(
            with_fsdp=False, with_mixed_precision=True, gpu_count=world_size
        )
        fsdp_config = cls._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=True, gpu_count=world_size
        )

        # Create fake inputs
        torch.random.manual_seed(gpu_id)
        batch_size = 1
        x = torch.randn(size=(batch_size, 384)).cuda()

        model_init_seed = 0
        args = {"in_dim": 384, "num_clusters": [65536], "normalize_last_layer": True}
        with set_torch_seed(model_init_seed):
            ddp_model = DINOHead(ddp_config["MODEL"], **args).cuda()
            ddp_model = DistributedDataParallel(ddp_model, device_ids=[gpu_id])
        with set_torch_seed(model_init_seed):
            fsdp_model = DINOHeadFSDP(fsdp_config["MODEL"], **args).cuda()
            print(fsdp_model)

        # Check that local_state_dict is valid
        fsdp_model_2 = DINOHeadFSDP(fsdp_config["MODEL"], **args).cuda()
        fsdp_model_2.load_local_state_dict(fsdp_model.local_state_dict())
        fsdp_model_2.load_state_dict(fsdp_model.state_dict())

        # DDP vs FSDP matching
        ddp_optimizer = optim.AdamW(ddp_model.parameters())
        fsdp_optimizer = optim.AdamW(fsdp_model.parameters())
        for _ in range(2):
            with torch.cuda.amp.autocast():
                ddp_loss = ddp_model(x)[0].sum()
                fsdp_loss = fsdp_model(x)[0].sum()
            ddp_optimizer.zero_grad()
            fsdp_optimizer.zero_grad()
            ddp_loss.backward()
            fsdp_loss.backward()
            ddp_optimizer.step()
            fsdp_optimizer.step()
            assert torch.allclose(ddp_loss, fsdp_loss)

    @gpu_test(gpu_count=2)
    def test_pretraining_dino_fp32(self) -> None:
        fsdp_config = self._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=False, with_multicrop=True, gpu_count=2
        )
        ddp_config = self._create_dino_pretraining_config(
            with_fsdp=False,
            with_mixed_precision=False,
            with_multicrop=True,
            gpu_count=2,
        )
        fsdp_losses = self.run_config(fsdp_config)
        ddp_losses = self.run_config(ddp_config)

        print(ddp_losses)
        print(fsdp_losses)
        self.assertGreater(len(fsdp_losses), 0)
        self.assertGreater(len(ddp_losses), 0)
        self.assertAlmostEqual(ddp_losses[0], fsdp_losses[0], places=4)
        self.assertAlmostEqual(ddp_losses[-1], fsdp_losses[-1], places=4)

    @gpu_test(gpu_count=2)
    def test_pretraining_dino_fp16(self) -> None:
        fsdp_config = self._create_dino_pretraining_config(
            with_fsdp=True, with_mixed_precision=True, with_multicrop=True, gpu_count=2
        )
        ddp_config = self._create_dino_pretraining_config(
            with_fsdp=False, with_mixed_precision=True, with_multicrop=True, gpu_count=2
        )
        fsdp_losses = self.run_config(fsdp_config)
        ddp_losses = self.run_config(ddp_config)

        print(ddp_losses)
        print(fsdp_losses)
        self.assertGreater(len(fsdp_losses), 0)
        self.assertGreater(len(ddp_losses), 0)
        self.assertAlmostEqual(ddp_losses[0], fsdp_losses[0], places=4)
        self.assertAlmostEqual(ddp_losses[-1], fsdp_losses[-1], places=4)

    @gpu_test(gpu_count=2)
    def test_pretraining_with_activation_checkpointing(self) -> None:
        config_ref = self._create_dino_pretraining_config(
            with_fsdp=True,
            with_mixed_precision=True,
            with_multicrop=True,
            gpu_count=2,
        )
        config_mlp = self._create_dino_pretraining_config(
            with_fsdp=True,
            with_mixed_precision=True,
            with_multicrop=True,
            gpu_count=2,
            with_mlp_checkpoints=True,
        )
        config_blk = self._create_dino_pretraining_config(
            with_fsdp=True,
            with_mixed_precision=True,
            with_multicrop=True,
            gpu_count=2,
            with_block_checkpoints=True,
        )
        losses_ref = self.run_config(config_ref)
        losses_mlp = self.run_config(config_mlp)
        losses_blk = self.run_config(config_blk)
        print(losses_ref)
        print(losses_mlp)
        print(losses_blk)
        self.assertAlmostEqual(losses_ref[-1], losses_mlp[-1], places=4)
        self.assertAlmostEqual(losses_ref[-1], losses_blk[-1], places=4)

    @gpu_test(gpu_count=2)
    def test_fsdp_teacher_checkpoint(self) -> None:
        with in_temporary_directory() as pretrain_dir:
            config = self._create_dino_pretraining_config(
                with_fsdp=True,
                with_mixed_precision=True,
                with_multicrop=True,
                gpu_count=2,
                num_epochs=1,
            )
            result = run_integration_test(config)
            print(result.get_losses())

            # Extract the teacher from the checkpoint
            DINOCheckpointUtils.extract_teacher_from_sharded_checkpoint(
                "model_final_checkpoint_phase0.torch",
                "model_final_checkpoint_phase0_teacher.torch",
            )
            folder_content = os.listdir(pretrain_dir)
            print(folder_content)
            self.assertIn("model_final_checkpoint_phase0_teacher.torch", folder_content)
            self.assertIn(
                "model_final_checkpoint_phase0_teacher_shard0.torch", folder_content
            )
            self.assertIn(
                "model_final_checkpoint_phase0_teacher_shard1.torch", folder_content
            )

            # Consolidate the checkpoint
            CheckpointFormatConverter.sharded_to_consolidated_checkpoint(
                "model_final_checkpoint_phase0_teacher.torch",
                "model_final_checkpoint_phase0_teacher_conso.torch",
            )
            folder_content = os.listdir(pretrain_dir)
            print(folder_content)
            self.assertIn(
                "model_final_checkpoint_phase0_teacher_conso.torch", folder_content
            )

            # Evaluate the checkpoint once consolidated
            eval_config = self._create_dino_linear_eval_config(
                checkpoint_path=os.path.join(
                    pretrain_dir, "model_final_checkpoint_phase0_teacher_conso.torch"
                ),
                gpu_count=2,
                with_fsdp=True,
            )
            eval_losses = self.run_config(eval_config)
            print(eval_losses)

    @gpu_test(gpu_count=2)
    def test_prehemption_during_training(self) -> None:
        with in_temporary_directory() as temp_dir:
            config = self._create_dino_pretraining_config(
                with_fsdp=True,
                with_mixed_precision=True,
                with_multicrop=True,
                gpu_count=2,
                num_epochs=2,
            )
            result = run_integration_test(config)
            losses_before = result.get_losses()

            temp_dir_content = os.listdir(temp_dir)
            self.assertIn("model_final_checkpoint_phase1.torch", temp_dir_content)
            os.remove("model_final_checkpoint_phase1.torch")
            os.remove("model_final_checkpoint_phase1_shard0.torch")
            os.remove("model_final_checkpoint_phase1_shard1.torch")
            os.remove("checkpoint.torch")
            os.remove("log.txt")

            result = run_integration_test(config)
            losses_after = result.get_losses()
            print(losses_before)
            print(losses_after)
            self.assertAlmostEqual(losses_after[-1], losses_before[-1], places=5)
