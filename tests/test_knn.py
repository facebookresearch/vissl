# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

import torch
from vissl.utils.checkpoint import CheckpointFormatConverter
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.knn_utils import run_knn_at_layer, run_knn_at_layer_low_memory
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestKNNBenchmark(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(with_fsdp: bool):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav_2crops",
                "+config/test/integration_test/models=swav_regnet_fsdp",
                "config.MODEL.FSDP_CONFIG.mixed_precision=False",
                "config.SEED_VALUE=0",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.LOSS.swav_loss.epsilon=0.03",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.DATA_LIMIT=32",
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
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        return config

    @staticmethod
    def _create_extract_features_config(
        checkpoint_path: str, model_name: str, with_fsdp: bool, num_gpu: int = 2
    ):
        cfg = compose_hydra_configuration(
            [
                "config=feature_extraction/extract_resnet_in1k_8gpu",
                "+config/test/integration_test/models=" + model_name,
                "config.MODEL.FSDP_CONFIG.mixed_precision=False",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_LABELS=10",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.RANDOM_SYNTHETIC_LABELS=10",
                "config.DATA.TRAIN.DATA_LIMIT=200",
                "config.DATA.TEST.DATA_LIMIT=200",
                "config.SEED_VALUE=0",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=20",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=10",
                "config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False",
                "config.EXTRACT_FEATURES.CHUNK_THRESHOLD=50",
                # Options used for the nearest neighbors config
                "config.NEAREST_NEIGHBOR.TOPK=20",
                "config.NEAREST_NEIGHBOR.SIGMA=0.1",
                "config.NEAREST_NEIGHBOR.L2_NORM_FEATS=True",
                "config.NEAREST_NEIGHBOR.USE_CUDA=False",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
        config.MODEL.TRUNK.REGNET.stage_checkpoints = [[2], [4], [6, 11], []]
        return config

    @classmethod
    def _create_extract_features_config_trunk(
        cls, checkpoint_path: str, with_fsdp: bool, num_gpu: int = 2
    ):
        return cls._create_extract_features_config(
            checkpoint_path,
            model_name="extraction_regnet_fsdp",
            with_fsdp=with_fsdp,
            num_gpu=num_gpu,
        )

    @classmethod
    def _create_extract_features_config_head(
        cls, checkpoint_path: str, with_fsdp: bool, num_gpu: int = 2
    ):
        return cls._create_extract_features_config(
            checkpoint_path,
            model_name="extraction_regnet_fsdp_head",
            with_fsdp=with_fsdp,
            num_gpu=num_gpu,
        )

    @classmethod
    def _create_extract_features_config_mid_head(
        cls, checkpoint_path: str, with_fsdp: bool, num_gpu: int = 2
    ):
        return cls._create_extract_features_config(
            checkpoint_path,
            model_name="extraction_regnet_fsdp_mid_head",
            with_fsdp=with_fsdp,
            num_gpu=num_gpu,
        )

    @gpu_test(gpu_count=2)
    def test_knn_fsdp(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_pretraining_config(with_fsdp=True)
            results = run_integration_test(pretrain_config)
            losses = results.get_losses()
            print(losses)

            # Convert checkpoint to sliced checkpoint for easy loading
            CheckpointFormatConverter.sharded_to_sliced_checkpoint(
                "checkpoint.torch", "checkpoint_sliced.torch"
            )
            checkpoint_path = os.path.join(pretrain_dir, "checkpoint_sliced.torch")

            # Create a directory to contain the extracted features
            with in_temporary_directory() as extract_dir:

                # Extract head features
                extract_config_head = self._create_extract_features_config_head(
                    checkpoint_path=checkpoint_path, with_fsdp=True
                )
                extract_config_head.EXTRACT_FEATURES.OUTPUT_DIR = extract_dir
                run_integration_test(
                    extract_config_head, engine_name="extract_features"
                )

                # Extract trunk features
                extract_config_trunk = self._create_extract_features_config_trunk(
                    checkpoint_path=checkpoint_path, with_fsdp=True
                )
                extract_config_trunk.EXTRACT_FEATURES.OUTPUT_DIR = extract_dir
                run_integration_test(
                    extract_config_trunk, engine_name="extract_features"
                )

                # Verify that we can merge the heads features back
                train_feat = ExtractedFeaturesLoader.load_features(
                    extract_dir, "train", "heads", flatten_features=True
                )
                self.assertEqual(train_feat["features"].shape, torch.Size([200, 128]))
                self.assertEqual(train_feat["targets"].shape, torch.Size([200, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([200]))

                # Verify that we can merge the trunk features back
                train_feat = ExtractedFeaturesLoader.load_features(
                    extract_dir, "train", "res5", flatten_features=True
                )
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([200, 3024 * 2 * 2])
                )
                self.assertEqual(train_feat["targets"].shape, torch.Size([200, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([200]))

                # Run KNN on the res5 layer
                extract_config_trunk.NEAREST_NEIGHBOR.FEATURES.PATH = extract_dir
                top_1_ref, top_5_ref, total_ref = run_knn_at_layer(
                    extract_config_trunk, layer_name="res5"
                )
                top_1_opt, top_5_opt, total_opt = run_knn_at_layer_low_memory(
                    extract_config_trunk, layer_name="res5"
                )
                self.assertEqual(total_ref, total_opt)
                # TODO - investigate: both KNN implementation have a bit of randomness
                #  in their accuracies, so the asserts are inequalities.
                self.assertLessEqual(top_1_ref, 30.0)
                self.assertLessEqual(top_1_opt, 30.0)
                self.assertGreaterEqual(top_1_ref, 29.0)
                self.assertGreaterEqual(top_1_opt, 29.0)
                # self.assertEqual(top_1_ref, top_1_opt)
                # self.assertEqual(top_5_ref, top_5_opt)

                # Run KNN on the head layer
                extract_config_head.NEAREST_NEIGHBOR.FEATURES.PATH = extract_dir
                top_1_ref, top_5_ref, total_ref = run_knn_at_layer(
                    extract_config_head, layer_name="heads"
                )
                top_1_opt, top_5_opt, total_opt = run_knn_at_layer_low_memory(
                    extract_config_head, layer_name="heads"
                )
                self.assertEqual(total_ref, total_opt)
                # TODO - investigate: both KNN implementation have a bit of randomness
                #  in their accuracies, so the asserts are inequalities.
                self.assertLessEqual(top_1_ref, 35.0)
                self.assertLessEqual(top_1_opt, 35.0)
                self.assertGreaterEqual(top_1_ref, 33.0)
                self.assertGreaterEqual(top_1_opt, 33.0)
                # self.assertEqual(top_1_ref, top_1_opt)
                # self.assertEqual(top_5_ref, top_5_opt)
