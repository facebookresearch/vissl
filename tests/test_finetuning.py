# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

from classy_vision.optim import build_optimizer_schedulers
from hydra.experimental import compose, initialize_config_module
from vissl.models import build_model
from vissl.optimizers import get_optimizer_param_groups
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    init_distributed_on_file,
    run_integration_test,
    with_temp_files,
)


class TestFineTuning(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(num_gpu: int = 2, with_fsdp: bool = False):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=test/integration_test/quick_swav",
                    "+config/test/integration_test/models=swav_regnet_fsdp",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.DATA_LIMIT=40",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                    "config.SEED_VALUE=0",
                    "config.LOSS.swav_loss.epsilon=0.03",
                    f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                    "config.LOG_FREQUENCY=1",
                    "config.OPTIMIZER.construct_single_param_group_only=True",
                ],
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
    def _create_finetuning_config(
        checkpoint_path: str,
        num_gpu: int = 2,
        regularize_bias: bool = False,
        construct_single_param_group_only: bool = False,
        with_fsdp: bool = False,
    ):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=test/integration_test/quick_eval_finetune_in1k",
                    "+config/test/integration_test/models=finetune_regnet_fsdp",
                    f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                    "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                    "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.DATA_LIMIT=40",
                    "config.DATA.TEST.DATA_LIMIT=20",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                    "config.DATA.TEST.BATCHSIZE_PER_REPLICA=2",
                    "config.SEED_VALUE=0",
                    f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                    "config.LOG_FREQUENCY=1",
                    "config.OPTIMIZER.num_epochs=2",
                    "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_value=0.01",
                    "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=2",
                    "config.OPTIMIZER.param_schedulers.lr_head.auto_lr_scaling.base_value=0.1",
                    "config.OPTIMIZER.param_schedulers.lr_head.auto_lr_scaling.base_lr_batch_size=2",
                    f"config.OPTIMIZER.regularize_bias={regularize_bias}",
                    f"config.OPTIMIZER.construct_single_param_group_only={construct_single_param_group_only}",
                ],
            )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "mlp_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "mlp"
        return config

    @gpu_test(gpu_count=1)
    def test_get_optimizer_param_groups(self):
        finetune_config = self._create_finetuning_config(
            checkpoint_path="",
            construct_single_param_group_only=False,
            regularize_bias=False,
        )
        optimizer_schedulers = build_optimizer_schedulers(finetune_config["OPTIMIZER"])
        base_model = build_model(finetune_config["MODEL"], finetune_config["OPTIMIZER"])
        param_groups = get_optimizer_param_groups(
            model=base_model,
            model_config=finetune_config["MODEL"],
            optimizer_config=finetune_config["OPTIMIZER"],
            optimizer_schedulers=optimizer_schedulers,
        )

        expected_param_groups = [
            {
                "params_count": 95,
                "params_numel": 80_419_552,
                "start_lr": 0.04,
                "end_lr": 0.0004,
                "weight_decay": 0.0001,
            },
            {
                "params_count": 154,
                "params_numel": 145_588,
                "start_lr": 0.04,
                "end_lr": 0.0004,
                "weight_decay": 0.0,
            },
            {
                # Params for linear layer matrix
                "params_count": 1,
                "params_numel": 3024 * 1000,
                "start_lr": 0.4,
                "end_lr": 0.004,
                "weight_decay": 1e-6,
            },
            {
                # Params for linear layer biases
                "params_count": 1,
                "params_numel": 1000,
                "start_lr": 0.4,
                "end_lr": 0.004,
                "weight_decay": 0.0,
            },
        ]

        for i, param_group in enumerate(param_groups):
            numel = sum(p.numel() for p in param_group["params"])
            self.assertEqual(set(param_group.keys()), {"params", "lr", "weight_decay"})
            self.assertEqual(
                len(param_group["params"]), expected_param_groups[i]["params_count"]
            )
            self.assertEqual(numel, expected_param_groups[i]["params_numel"])
            self.assertEqual(
                param_group["lr"]._start_value, expected_param_groups[i]["start_lr"]
            )
            self.assertEqual(
                param_group["lr"]._end_value, expected_param_groups[i]["end_lr"]
            )
            self.assertEqual(
                param_group["weight_decay"], expected_param_groups[i]["weight_decay"]
            )

    @gpu_test(gpu_count=1)
    def test_get_optimizer_param_groups_fsdp_single_group(self):
        with with_temp_files(count=1) as sync_file:
            init_distributed_on_file(world_size=1, gpu_id=0, sync_file=sync_file)

            finetune_config = self._create_finetuning_config(
                checkpoint_path="",
                construct_single_param_group_only=True,
                regularize_bias=False,
                with_fsdp=True,
            )
            optimizer_schedulers = build_optimizer_schedulers(
                finetune_config["OPTIMIZER"]
            )
            base_model = build_model(
                finetune_config["MODEL"], finetune_config["OPTIMIZER"]
            )
            param_groups = get_optimizer_param_groups(
                model=base_model,
                model_config=finetune_config["MODEL"],
                optimizer_config=finetune_config["OPTIMIZER"],
                optimizer_schedulers=optimizer_schedulers,
            )

            expected_param_groups = [{"param_numel": 83590140}]

            for i, param_group in enumerate(param_groups):
                numel = sum(p.numel() for p in param_group["params"])
                self.assertEqual(expected_param_groups[i]["param_numel"], numel)

    @gpu_test(gpu_count=2)
    def test_fine_tuning_end_to_end(self):
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_pretraining_config()
            run_integration_test(pretrain_config)
            checkpoint_path = os.path.join(pretrain_dir, "checkpoint.torch")

            # Create a separate directly in which to run the fine-tuning
            with in_temporary_directory():
                finetune_config = self._create_finetuning_config(
                    checkpoint_path,
                    construct_single_param_group_only=False,
                    regularize_bias=False,
                )
                result = run_integration_test(finetune_config)
                accuracies = result.get_accuracies(from_metrics_file=True)
                self.assertEqual(4, len(accuracies))
