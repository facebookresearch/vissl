# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

from classy_vision.optim import build_optimizer_schedulers
from vissl.config import AttrDict
from vissl.models import build_model
from vissl.optimizers import get_optimizer_param_groups
from vissl.utils.checkpoint import CheckpointFormatConverter
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
    spawn_distributed_test,
)


class TestFineTuning(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(
        num_gpu: int = 2, with_fsdp: bool = False, fsdp_flatten_parameters: bool = False
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "+config/test/integration_test/models=swav_regnet_fsdp",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.SEED_VALUE=0",
                "config.LOSS.swav_loss.epsilon=0.03",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
                "config.OPTIMIZER.construct_single_param_group_only=True",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.flatten_parameters = fsdp_flatten_parameters
            config.MODEL.FSDP_CONFIG.mixed_precision = False
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
        fsdp_flatten_parameters: bool = False,
        with_partial_head: bool = False,
    ):
        architecture_config = (
            "+config/test/integration_test/models=finetune_regnet_fsdp"
        )
        if with_partial_head:
            architecture_config = (
                "+config/test/integration_test/models=finetune_regnet_fsdp_head"
            )

        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_finetune_in1k",
                architecture_config,
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
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
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            if with_partial_head:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
                config["MODEL"]["HEAD"]["PARAMS"][1][0] = "mlp_fsdp"
            else:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "mlp_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.flatten_parameters = fsdp_flatten_parameters
            config.MODEL.FSDP_CONFIG.mixed_precision = False
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            if with_partial_head:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
                config["MODEL"]["HEAD"]["PARAMS"][1][0] = "mlp"
            else:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "mlp"
        return config

    @staticmethod
    def _expected_finetuning_param_groups(
        construct_single_param_group_only: bool = False, sharding_factor: int = 1
    ):
        if construct_single_param_group_only:
            return [
                {
                    "params_count": 138,
                    "params_numel": 83_590_140 // sharding_factor,
                    "start_lr": 0.04,
                    "end_lr": 0.0004,
                    "weight_decay": 0.0001,
                }
            ]
        else:
            return [
                {
                    "params_count": 95,
                    "params_numel": 80_419_552 // sharding_factor,
                    "start_lr": 0.04,
                    "end_lr": 0.0004,
                    "weight_decay": 0.0001,
                },
                {
                    "params_count": 154,
                    "params_numel": 145_588 // sharding_factor,
                    "start_lr": 0.04,
                    "end_lr": 0.0004,
                    "weight_decay": 0.0,
                },
                {
                    # Params for linear layer matrix
                    "params_count": 1,
                    "params_numel": 3024 * 1000 // sharding_factor,
                    "start_lr": 0.4,
                    "end_lr": 0.004,
                    "weight_decay": 1e-6,
                },
                {
                    # Params for linear layer biases
                    "params_count": 1,
                    "params_numel": 1000 // sharding_factor,
                    "start_lr": 0.4,
                    "end_lr": 0.004,
                    "weight_decay": 0.0,
                },
            ]

    @staticmethod
    def _compute_param_groups(finetune_config: AttrDict):
        """
        Take a configuration and compute the parameter groups
        for this configuration
        """
        optimizer_schedulers = build_optimizer_schedulers(finetune_config["OPTIMIZER"])
        base_model = build_model(finetune_config["MODEL"], finetune_config["OPTIMIZER"])
        return get_optimizer_param_groups(
            model=base_model,
            model_config=finetune_config["MODEL"],
            optimizer_config=finetune_config["OPTIMIZER"],
            optimizer_schedulers=optimizer_schedulers,
        )

    @staticmethod
    def _assert_equal(a, b):
        if a != b:
            raise AssertionError(f"Expected {a} == {b}")

    @classmethod
    def _check_valid_param_groups(cls, expected_param_groups, param_groups):
        for i, param_group in enumerate(param_groups):
            numel = sum(p.numel() for p in param_group["params"])
            cls._assert_equal(set(param_group.keys()), {"params", "lr", "weight_decay"})
            cls._assert_equal(
                len(param_group["params"]), expected_param_groups[i]["params_count"]
            )
            cls._assert_equal(numel, expected_param_groups[i]["params_numel"])
            cls._assert_equal(
                param_group["lr"]._start_value, expected_param_groups[i]["start_lr"]
            )
            cls._assert_equal(
                param_group["lr"]._end_value, expected_param_groups[i]["end_lr"]
            )
            cls._assert_equal(
                param_group["weight_decay"], expected_param_groups[i]["weight_decay"]
            )

    @gpu_test(gpu_count=1)
    def test_get_optimizer_param_groups(self) -> None:
        finetune_config = self._create_finetuning_config(
            checkpoint_path="",
            construct_single_param_group_only=False,
            regularize_bias=False,
        )
        expected_param_groups = self._expected_finetuning_param_groups(
            construct_single_param_group_only=False
        )
        param_groups = self._compute_param_groups(finetune_config)
        self._check_valid_param_groups(expected_param_groups, param_groups)

    @staticmethod
    def _test_get_optimizer_param_groups_fsdp_single_group_worker(gpu_id: int):
        finetune_config = TestFineTuning._create_finetuning_config(
            checkpoint_path="",
            construct_single_param_group_only=True,
            regularize_bias=False,
            with_fsdp=True,
            fsdp_flatten_parameters=True,
        )
        expected_param_groups = TestFineTuning._expected_finetuning_param_groups(
            construct_single_param_group_only=True, sharding_factor=2
        )
        param_groups = TestFineTuning._compute_param_groups(finetune_config)
        TestFineTuning._check_valid_param_groups(expected_param_groups, param_groups)

    @gpu_test(gpu_count=2)
    def test_get_optimizer_param_groups_fsdp_single_group(self) -> None:
        spawn_distributed_test(
            gpu_count=2,
            worker_fn=self._test_get_optimizer_param_groups_fsdp_single_group_worker,
        )

    @staticmethod
    def _test_get_optimizer_param_groups_fsdp_worker(gpu_id: int):
        finetune_config = TestFineTuning._create_finetuning_config(
            checkpoint_path="",
            construct_single_param_group_only=False,
            regularize_bias=False,
            with_fsdp=True,
            fsdp_flatten_parameters=False,
        )
        expected_param_groups = TestFineTuning._expected_finetuning_param_groups(
            construct_single_param_group_only=False, sharding_factor=2
        )
        param_groups = TestFineTuning._compute_param_groups(finetune_config)
        TestFineTuning._check_valid_param_groups(expected_param_groups, param_groups)

    @gpu_test(gpu_count=2)
    def test_get_optimizer_param_groups_fsdp(self) -> None:
        spawn_distributed_test(
            gpu_count=2, worker_fn=self._test_get_optimizer_param_groups_fsdp_worker
        )

    @gpu_test(gpu_count=2)
    def test_fine_tuning_end_to_end(self) -> None:
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

    @gpu_test(gpu_count=2)
    def test_fine_tuning_end_to_end_fsdp(self) -> None:
        with in_temporary_directory() as pretrain_dir:
            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_pretraining_config(
                with_fsdp=True, fsdp_flatten_parameters=True
            )
            run_integration_test(pretrain_config)
            sharded_checkpoint_path = os.path.join(pretrain_dir, "checkpoint.torch")

            # Consolidate the checkpoint of the FSDP model
            conso_checkpoint_path = os.path.join(pretrain_dir, "consolidated.torch")
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                input_checkpoint_path=sharded_checkpoint_path,
                output_checkpoint_path=conso_checkpoint_path,
            )

            # Consolidate the checkpoint of the FSDP model (sliced version)
            sliced_checkpoint_path = os.path.join(pretrain_dir, "sliced.torch")
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                input_checkpoint_path=sharded_checkpoint_path,
                output_checkpoint_path=sliced_checkpoint_path,
            )

            # Create a separate directory in which to run the fine-tuning
            with in_temporary_directory():
                finetune_config = self._create_finetuning_config(
                    sliced_checkpoint_path,
                    construct_single_param_group_only=False,
                    regularize_bias=False,
                    with_fsdp=True,
                    fsdp_flatten_parameters=False,
                    with_partial_head=False,
                )
                result = run_integration_test(finetune_config)
                accuracies = result.get_accuracies(from_metrics_file=True)
                self.assertEqual(4, len(accuracies))

            # Create a separate directory in which we run the fine-tuning
            # with a partial head loading (sliced checkpoint)
            with in_temporary_directory():
                finetune_config = self._create_finetuning_config(
                    sliced_checkpoint_path,
                    construct_single_param_group_only=False,
                    regularize_bias=False,
                    with_fsdp=True,
                    fsdp_flatten_parameters=False,
                    with_partial_head=True,
                )
                result = run_integration_test(finetune_config)
                losses = result.get_losses()
                first_loss_sliced = losses[0]
                accuracies = result.get_accuracies(from_metrics_file=True)
                self.assertEqual(4, len(accuracies))

            # Create a separate directory in which we run the fine-tuning
            # with a partial head loading (consolidated checkpoint)
            with in_temporary_directory():
                finetune_config = self._create_finetuning_config(
                    conso_checkpoint_path,
                    construct_single_param_group_only=False,
                    regularize_bias=False,
                    with_fsdp=True,
                    fsdp_flatten_parameters=False,
                    with_partial_head=True,
                )
                result = run_integration_test(finetune_config)
                losses = result.get_losses()
                self.assertAlmostEqual(first_loss_sliced, losses[0], places=4)
                accuracies = result.get_accuracies(from_metrics_file=True)
                self.assertEqual(4, len(accuracies))
