# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

import torch
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.misc import merge_features
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestExtractClusterWorkflow(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.SEED_VALUE=0",
                "config.MODEL.AMP_PARAMS.USE_AMP=False",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.LOSS.swav_loss.epsilon=0.03",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.MODEL.FSDP_CONFIG.mixed_precision=False",
                "config.MODEL.FSDP_CONFIG.fp32_reduce_scatter=False",
                "config.MODEL.FSDP_CONFIG.compute_dtype=float32",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.OPTIMIZER.use_larc=False",
                "config.EXTRACT_FEATURES.CHUNK_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_extract_features_config_head(checkpoint_path: str, num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=feature_extraction/extract_resnet_in1k_8gpu",
                "+config/feature_extraction/with_head=rn50_swav",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TEST.DATA_LIMIT=20",
                "config.SEED_VALUE=0",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=2",
                "config.EXTRACT_FEATURES.CHUNK_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_extract_features_config_trunk(checkpoint_path: str, num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=feature_extraction/extract_resnet_in1k_8gpu",
                "+config/feature_extraction/trunk_only=rn50_layers",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TEST.DATA_LIMIT=20",
                "config.SEED_VALUE=0",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=2",
                "config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False",
                "config.EXTRACT_FEATURES.CHUNK_THRESHOLD=0",
            ],
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_extract_cluster_assignment_ddp(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_pretraining_config()
            run_integration_test(pretrain_config)

            # Create a directory to contain the extracted features
            with in_temporary_directory() as extract_dir:

                # Run the extract engine in a separate directory to check that
                # it is correctly able to output the feature in a another dir
                with in_temporary_directory():
                    extract_config = self._create_extract_features_config_head(
                        checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch")
                    )
                    extract_config.EXTRACT_FEATURES.OUTPUT_DIR = extract_dir
                    run_integration_test(extract_config, engine_name="extract_features")

                # Check the content of the directory containing the extracted dirs
                folder_content = os.listdir(extract_dir)
                print(folder_content)
                for rank in [0, 1]:
                    for chunk in range(5):
                        for file in [
                            f"rank{rank}_chunk{chunk}_train_heads_features.npy",
                            f"rank{rank}_chunk{chunk}_train_heads_inds.npy",
                            f"rank{rank}_chunk{chunk}_train_heads_targets.npy",
                        ]:
                            self.assertIn(file, folder_content)

                # Verify that we can merge the features back (train split)
                train_feat = merge_features(extract_dir, "train", "heads")
                print(train_feat)
                self.assertEqual(train_feat["features"].shape, torch.Size([40, 128]))
                self.assertEqual(train_feat["targets"].shape, torch.Size([40, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([40]))

                # Verify that we can merge the features back (test split)
                test_feat = merge_features(extract_dir, "test", "heads")
                self.assertEqual(test_feat["features"].shape, torch.Size([20, 128]))
                self.assertEqual(test_feat["targets"].shape, torch.Size([20, 1]))
                self.assertEqual(test_feat["inds"].shape, torch.Size([20]))

                # Run the extract engine this time for the features of the trunk
                with in_temporary_directory():
                    extract_config = self._create_extract_features_config_trunk(
                        checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch")
                    )
                    extract_config.EXTRACT_FEATURES.OUTPUT_DIR = extract_dir
                    run_integration_test(extract_config, engine_name="extract_features")

                # Verify that we can merge the features back without flattening them
                train_feat = merge_features(extract_dir, "train", "res5")
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([40, 2048, 2, 2])
                )
                self.assertEqual(train_feat["targets"].shape, torch.Size([40, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([40]))

                # Verify that we can merge the features back without
                # flattening them (second approach)
                train_feat = ExtractedFeaturesLoader.load_features(
                    extract_dir, "train", "res5"
                )
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([40, 2048, 2, 2])
                )

                # Verify that we can merge the features back but flattened
                train_feat = ExtractedFeaturesLoader.load_features(
                    extract_dir, "train", "res5", flatten_features=True
                )
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([40, 2048 * 2 * 2])
                )
                self.assertEqual(train_feat["targets"].shape, torch.Size([40, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([40]))

                # Verify that we can sample the features (unflattened)
                train_feat = ExtractedFeaturesLoader.sample_features(
                    input_dir=extract_dir,
                    split="train",
                    layer="res5",
                    num_samples=10,
                    seed=0,
                )
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([10, 2048, 2, 2])
                )
                self.assertEqual(train_feat["targets"].shape, torch.Size([10, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([10]))

                # Verify that we can sample the features (flattened)
                train_feat = ExtractedFeaturesLoader.sample_features(
                    input_dir=extract_dir,
                    split="train",
                    layer="res5",
                    num_samples=10,
                    seed=0,
                    flatten_features=True,
                )
                self.assertEqual(
                    train_feat["features"].shape, torch.Size([10, 2048 * 2 * 2])
                )
                self.assertEqual(train_feat["targets"].shape, torch.Size([10, 1]))
                self.assertEqual(train_feat["inds"].shape, torch.Size([10]))
