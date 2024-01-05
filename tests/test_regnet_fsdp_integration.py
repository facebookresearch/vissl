# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

import numpy as np
import torch
from vissl.utils.checkpoint import CheckpointFormatConverter
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestRegnetFSDPIntegration(unittest.TestCase):
    """
    Test the Regnet FSDP model in comparison with the DDP Regnet
    to verify that both converge to the same losses
    """

    @staticmethod
    def _create_pretraining_config(
        with_fsdp: bool,
        with_activation_checkpointing: bool,
        with_mixed_precision: bool,
        auto_wrap_threshold: int,
        force_sync_all_gather: bool = False,
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav_2crops",
                "+config/test/integration_test/models=swav_regnet_fsdp",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.LOSS.swav_loss.epsilon=0.03",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.OPTIMIZER.use_larc=False",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
                f"config.MODEL.FSDP_CONFIG.FORCE_SYNC_CUDA={force_sync_all_gather}",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD = auto_wrap_threshold
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = (
            with_activation_checkpointing
        )
        return config

    def _create_linear_evaluation_config(
        self, with_fsdp: bool, with_mixed_precision: bool, auto_wrap_threshold: int
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_in1k_linear",
                "+config/test/integration_test/models=eval_regnet_fsdp",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.OPTIMIZER.use_larc=False",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "eval_mlp_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD = auto_wrap_threshold
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "eval_mlp"
        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = False
        return config

    @staticmethod
    def _create_finetuning_config(
        checkpoint_path: str,
        auto_wrap_threshold: int,
        with_fsdp: bool = False,
        with_partial_head: bool = False,
        with_mixed_precision: bool = False,
        with_activation_checkpointing: bool = False,
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
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=2",
                "config.SEED_VALUE=0",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
                "config.LOG_FREQUENCY=1",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.OPTIMIZER.num_epochs=2",
                "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_value=0.01",
                "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=2",
                "config.OPTIMIZER.param_schedulers.lr_head.auto_lr_scaling.base_value=0.1",
                "config.OPTIMIZER.param_schedulers.lr_head.auto_lr_scaling.base_lr_batch_size=2",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
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
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD = auto_wrap_threshold
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            if with_partial_head:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
                config["MODEL"]["HEAD"]["PARAMS"][1][0] = "mlp"
            else:
                config["MODEL"]["HEAD"]["PARAMS"][0][0] = "mlp"

        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = (
            with_activation_checkpointing
        )
        return config

    def _create_extract_label_prediction_finetuned_config(
        self, with_fsdp: bool, with_mixed_precision: bool, auto_wrap_threshold: int
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_extract_label_predictions",
                "+config/test/integration_test/models=extract_label_pred_regnet_fsdp_fine_tuned",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][1][0] = "mlp_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD = auto_wrap_threshold
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "eval_mlp"
        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = False
        return config

    def _create_extract_label_prediction_config(
        self, with_fsdp: bool, with_mixed_precision: bool, auto_wrap_threshold: int
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_extract_label_predictions",
                "+config/test/integration_test/models=extract_label_pred_regnet_fsdp",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_LABELS=2",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.OPTIMIZER.num_epochs=1",
                "config.OPTIMIZER.use_larc=False",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "eval_mlp_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
            config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD = auto_wrap_threshold
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "eval_mlp"
        config.MODEL.AMP_PARAMS.USE_AMP = with_mixed_precision
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        config.MODEL.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING = False
        return config

    def run_pretraining(
        self,
        with_fsdp: bool,
        with_activation_checkpointing: bool,
        with_mixed_precision: bool,
        auto_wrap_threshold: int = 0,
        force_sync_all_gather: bool = False,
    ):
        with in_temporary_directory():
            config = self._create_pretraining_config(
                with_fsdp=with_fsdp,
                with_activation_checkpointing=with_activation_checkpointing,
                with_mixed_precision=with_mixed_precision,
                auto_wrap_threshold=auto_wrap_threshold,
                force_sync_all_gather=force_sync_all_gather,
            )
            result = run_integration_test(config)
            return result.get_losses()

    def run_linear_eval(
        self,
        checkpoint_path: str,
        with_fsdp: bool,
        with_mixed_precision: bool,
        auto_wrap_threshold: int = 0,
    ):
        with in_temporary_directory():
            config = self._create_linear_evaluation_config(
                with_fsdp=with_fsdp,
                with_mixed_precision=with_mixed_precision,
                auto_wrap_threshold=auto_wrap_threshold,
            )
            config.MODEL.WEIGHTS_INIT.PARAMS_FILE = checkpoint_path
            result = run_integration_test(config)
            return result.get_losses()

    @gpu_test(gpu_count=2)
    def test_fsdp_integration(self) -> None:
        fsdp_losses_1 = self.run_pretraining(
            with_fsdp=True,
            with_activation_checkpointing=True,
            with_mixed_precision=False,
            force_sync_all_gather=True,
        )
        fsdp_losses_2 = self.run_pretraining(
            with_fsdp=True,
            with_activation_checkpointing=False,
            with_mixed_precision=False,
            force_sync_all_gather=False,
        )
        fsdp_losses_3 = self.run_pretraining(
            with_fsdp=True,
            with_activation_checkpointing=False,
            with_mixed_precision=False,
            auto_wrap_threshold=100,
            force_sync_all_gather=False,
        )
        ddp_losses = self.run_pretraining(
            with_fsdp=False,
            with_activation_checkpointing=False,
            with_mixed_precision=False,
        )
        self.assertEqual(ddp_losses, fsdp_losses_1)
        self.assertEqual(ddp_losses, fsdp_losses_2)
        self.assertEqual(ddp_losses, fsdp_losses_3)

    @gpu_test(gpu_count=2)
    def test_fsdp_integration_with_linear_eval(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Start pre-training
            config = self._create_pretraining_config(
                with_fsdp=True,
                with_activation_checkpointing=True,
                with_mixed_precision=False,
                auto_wrap_threshold=0,
            )
            run_integration_test(config)

            # Consolidate the weights (3 different ways)
            CheckpointFormatConverter.sharded_to_consolidated_checkpoint(
                "checkpoint.torch", "checkpoint_conso.torch"
            )
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                "checkpoint.torch", "checkpoint_sliced.torch"
            )
            CheckpointFormatConverter.consolidated_to_sliced_checkpoint(
                "checkpoint_conso.torch", "checkpoint_sliced_2.torch"
            )

            # Load the sharded checkpoint and perform a inear evaluation on it
            ref_losses = self.run_linear_eval(
                checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch"),
                with_fsdp=True,
                with_mixed_precision=False,
                auto_wrap_threshold=0,
            )
            self.assertEqual(8, len(ref_losses))

            # Then check that the results are the same for the other kind of
            # checkpoints after consolidation has taken place
            for checkpoint_name in [
                "checkpoint_conso.torch",
                "checkpoint_sliced.torch",
                "checkpoint_sliced_2.torch",
            ]:
                losses = self.run_linear_eval(
                    checkpoint_path=os.path.join(pretrain_dir, checkpoint_name),
                    with_fsdp=True,
                    with_mixed_precision=False,
                    auto_wrap_threshold=0,
                )
                self.assertEqual(8, len(losses))
                self.assertAlmostEqual(
                    losses[0],
                    ref_losses[0],
                    places=4,
                    msg=f"Failed for {checkpoint_name}",
                )

    @gpu_test(gpu_count=2)
    def test_fsdp_integration_mixed_precision(self) -> None:
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

    @gpu_test(gpu_count=2)
    def test_fsdp_extract_label_predictions(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Start pre-training and consolidate the checkpoint
            config = self._create_pretraining_config(
                with_fsdp=True,
                with_activation_checkpointing=True,
                with_mixed_precision=False,
                auto_wrap_threshold=0,
            )
            run_integration_test(config)
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                "checkpoint.torch", "checkpoint_sliced.torch"
            )

            with in_temporary_directory() as fine_tune_dir:

                # Run fine-tuning on the consolidated checkpoint
                finetune_config = self._create_finetuning_config(
                    checkpoint_path=os.path.join(
                        pretrain_dir, "checkpoint_sliced.torch"
                    ),
                    auto_wrap_threshold=True,
                    with_fsdp=True,
                    with_partial_head=True,
                    with_mixed_precision=True,
                    with_activation_checkpointing=True,
                )
                finetune_config.OPTIMIZER.num_epochs = 1
                results = run_integration_test(finetune_config)
                fine_tune_accuracy = results.get_final_accuracy(layer_name="0")

                # Consolidate the fine-tuned evaluation checkpoint
                CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                    "checkpoint.torch", "checkpoint_sliced.torch"
                )

                # Run label extract on top of fine-tuned checkpoint
                # Run label extraction on both sharded and consolidated checkpoint
                for checkpoint_name in ["checkpoint.torch", "checkpoint_sliced.torch"]:
                    with in_temporary_directory() as extract_dir:
                        extract_config = (
                            self._create_extract_label_prediction_finetuned_config(
                                with_fsdp=True,
                                with_mixed_precision=False,
                                auto_wrap_threshold=True,
                            )
                        )
                        extract_config.MODEL.WEIGHTS_INIT.PARAMS_FILE = os.path.join(
                            fine_tune_dir, checkpoint_name
                        )
                        run_integration_test(
                            extract_config, engine_name="extract_label_predictions"
                        )

                        accuracy, preds, targets = self._read_accuracy(
                            extract_dir, layer="heads"
                        )
                        print(targets)
                        print(preds)
                        print("Accuracy:", accuracy)
                        self.assertAlmostEqual(
                            accuracy * 100.0, fine_tune_accuracy, places=3
                        )

            with in_temporary_directory() as lin_eval_dir:

                # Run linear evaluation on the consolidated checkpoint
                config = self._create_linear_evaluation_config(
                    with_fsdp=True, with_mixed_precision=True, auto_wrap_threshold=True
                )
                config.OPTIMIZER.num_epochs = 1
                config.OPTIMIZER.param_schedulers.lr.values = [0.01]
                config.OPTIMIZER.param_schedulers.lr.milestones = []
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE = os.path.join(
                    pretrain_dir, "checkpoint_sliced.torch"
                )
                results = run_integration_test(config)
                linear_accuracy = results.get_final_accuracy("res5")

                # Consolidate the linear evaluation checkpoint
                CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                    "checkpoint.torch", "checkpoint_sliced.torch"
                )

                # Run label extraction on both sharded and consolidated checkpoint
                for checkpoint_name in ["checkpoint.torch", "checkpoint_sliced.torch"]:
                    with in_temporary_directory() as extract_dir:
                        extract_config = self._create_extract_label_prediction_config(
                            with_fsdp=True,
                            with_mixed_precision=False,
                            auto_wrap_threshold=True,
                        )
                        extract_config.MODEL.WEIGHTS_INIT.PARAMS_FILE = os.path.join(
                            lin_eval_dir, checkpoint_name
                        )
                        run_integration_test(
                            extract_config, engine_name="extract_label_predictions"
                        )

                        # Read the predictions and verify that they match
                        # the linear evaluation results
                        accuracy, preds, targets = self._read_accuracy(
                            extract_dir, layer="res5"
                        )
                        print(targets)
                        print(preds)
                        print("Accuracy:", accuracy)
                        self.assertAlmostEqual(
                            accuracy * 100.0, linear_accuracy, places=3
                        )

    def _read_accuracy(self, extract_dir: str, layer: str):
        print(os.listdir(extract_dir))
        targets, preds = [], []
        for file in sorted(os.listdir(extract_dir)):
            if f"test_{layer}_targets.npy" in file:
                targets.append(np.load(file))
                print(file)
            elif f"test_{layer}_predictions.npy" in file:
                preds.append(np.load(file))
                print(file)
        targets = np.squeeze(np.concatenate(targets, axis=0))
        preds = np.squeeze(np.concatenate(preds, axis=0))
        correct = (targets == preds).sum()
        accuracy = correct / targets.shape[0]
        return accuracy, preds, targets
