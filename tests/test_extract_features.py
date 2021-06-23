# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

from hydra.experimental import compose, initialize_config_module
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestExtractClusterWorkflow(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(num_gpu: int = 2):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
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
                ],
            )

        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_extract_features_config(checkpoint_path: str, num_gpu: int = 2):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=feature_extraction/extract_resnet_in1k_8gpu",
                    "+config/feature_extraction/with_head=rn50_swav",
                    f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                    "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                    "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.DATA_LIMIT=40",
                    "config.DATA.TEST.DATA_LIMIT=40",
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
                ],
            )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_extract_cluster_assignment_ddp(self):
        with in_temporary_directory() as pretrain_dir:

            pretrain_config = self._create_pretraining_config()
            run_integration_test(pretrain_config)

            with in_temporary_directory() as extract_dir:
                extract_config = self._create_extract_features_config(
                    checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch")
                )

                run_integration_test(extract_config, engine_name="extract_features")
                folder_content = os.listdir(extract_dir)
                print(folder_content)
                for rank in [0, 1]:
                    for feat_name in ["heads"]:
                        for file in [
                            f"rank{rank}_train_{feat_name}_features.npy",
                            f"rank{rank}_train_{feat_name}_inds.npy",
                            f"rank{rank}_train_{feat_name}_targets.npy",
                        ]:
                            self.assertIn(file, folder_content)
