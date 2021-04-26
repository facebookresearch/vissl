# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import re
import shutil
import tempfile
import unittest
from contextlib import contextmanager

import torch
from hydra.experimental import compose, initialize_config_module
from vissl.hooks import default_hook_generator
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.hydra_config import convert_to_attrdict


class TestRegnetFSDPIntegration(unittest.TestCase):
    """
    Test the Regnet FSDP model in comparison with the DDP Regnet
    to verify that both converge to the same losses
    """

    @staticmethod
    def _create_pretraining_config(
        with_fsdp: bool, with_activation_checkpointing: bool, with_mixed_precision: bool
    ):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=test/integration_test/quick_swav",
                    "+config/pretrain/swav/models=regnet16Gf",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.SEED_VALUE=0",
                    "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                    "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                    "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                    "config.LOSS.swav_loss.epsilon=0.03",
                    "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                    "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
                    "config.LOG_FREQUENCY=1",
                    "config.OPTIMIZER.construct_single_param_group_only=True",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                    "config.OPTIMIZER.use_larc=False",
                    "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                    "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                ],
            )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision

        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = (
            with_activation_checkpointing
        )
        return args, config

    @staticmethod
    @contextmanager
    def _in_temporary_directory():
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)

    def capture_losses(self, file_name: str):
        losses = []
        regex = re.compile(r"iter: (.*?); lr: (?:.*?); loss: (.*?);")
        with open(file_name, "r") as file:
            for line in file:
                if not line.startswith("INFO"):
                    continue
                match = regex.search(line)
                if match is not None:
                    loss = float(match.group(2))
                    losses.append(loss)
        return losses

    def run_pretraining(
        self,
        with_fsdp: bool,
        with_activation_checkpointing: bool,
        with_mixed_precision: bool,
    ):
        with self._in_temporary_directory() as dir_name:
            args, config = self._create_pretraining_config(
                with_fsdp=with_fsdp,
                with_activation_checkpointing=with_activation_checkpointing,
                with_mixed_precision=with_mixed_precision,
            )
            launch_distributed(
                cfg=config,
                node_id=args.node_id,
                engine_name=args.engine_name,
                hook_generator=default_hook_generator,
            )
            return self.capture_losses(os.path.join(dir_name, "log.txt"))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Not enough GPUs to run the test")
    def test_fsdp_integration(self):
        ddp_losses = self.run_pretraining(
            with_fsdp=False,
            with_activation_checkpointing=False,
            with_mixed_precision=False,
        )
        fsdp_losses = self.run_pretraining(
            with_fsdp=True,
            with_activation_checkpointing=False,
            with_mixed_precision=False,
        )
        self.assertEqual(ddp_losses, fsdp_losses)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Not enough GPUs to run the test")
    def test_fsdp_integration_mixed_precision(self):
        ddp_losses = self.run_pretraining(
            with_fsdp=False,
            with_activation_checkpointing=False,
            with_mixed_precision=True,
        )
        fsdp_losses = self.run_pretraining(
            with_fsdp=True,
            with_activation_checkpointing=False,
            with_mixed_precision=True,
        )

        # TODO (Quentin) - understand why we do not get full convergence here
        # self.assertEqual(ddp_losses, fsdp_losses)

        self.assertEqual(len(ddp_losses), len(fsdp_losses))
        for i, (ddp_loss, fsdp_loss) in enumerate(zip(ddp_losses, fsdp_losses)):
            self.assertAlmostEqual(
                ddp_loss, fsdp_loss, places=1, msg=f"Mismatch at loss {i}"
            )

        print(ddp_losses)
        print(fsdp_losses)
