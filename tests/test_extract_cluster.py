# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import unittest

import numpy as np
from vissl.utils.cluster_utils import ClusterAssignmentLoader
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestExtractClusterWorkflow(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(with_fsdp: bool, num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "+config/pretrain/swav/models=regnet16Gf",
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
    def _create_extract_cluster_config(
        with_fsdp: bool, checkpoint_path: str, num_gpu: int = 2
    ):
        cfg = compose_hydra_configuration(
            [
                "config=extract_cluster/swav/visualise_swav_resnet_in1k_8gpu",
                "+config/extract_cluster/swav/models=regnet16Gf",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TEST.DATA_LIMIT=20",
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
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.OPTIMIZER.use_larc=False",
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

    def run_cluster_assignment(self, with_fsdp: bool):
        with in_temporary_directory() as pretrain_dir:

            # Pre-train a SwAV model in order to get some weights
            pretrain_config = self._create_pretraining_config(with_fsdp=with_fsdp)
            run_integration_test(pretrain_config)

            # Extract the cluster assignments of each sample
            with in_temporary_directory() as extract_dir:
                extract_config = self._create_extract_cluster_config(
                    with_fsdp=with_fsdp,
                    checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch"),
                )
                extract_config.EXTRACT_FEATURES.CHUNK_THRESHOLD = 10
                run_integration_test(extract_config, engine_name="extract_cluster")
                extraction_outputs = os.listdir(extract_dir)

                # Check that the cluster assignments are computed in both
                # compact format and dataset disk_filelist format
                self.assertIn("cluster_assignments.torch", extraction_outputs)
                self.assertIn("train_images.npy", extraction_outputs)
                self.assertIn("train_labels.npy", extraction_outputs)
                self.assertIn("test_images.npy", extraction_outputs)
                self.assertIn("test_labels.npy", extraction_outputs)

                # Check that the soft assignments (on prototypes) are exported
                for rank in range(2):
                    for chunk in range(2):
                        file_name = f"rank{rank}_chunk{chunk}_train_heads_protos.npy"
                        self.assertIn(file_name, extraction_outputs)
                        self.assertEqual(np.load(file_name).shape[1], 3000)
                    file_name = f"rank{rank}_chunk0_test_heads_protos.npy"
                    self.assertIn(file_name, extraction_outputs)
                    self.assertEqual(np.load(file_name).shape[1], 3000)

                # Copy the cluster assignments
                shutil.move(
                    src=os.path.join(extract_dir, "cluster_assignments.torch"),
                    dst=os.path.join(pretrain_dir, "cluster_assignments.torch"),
                )

            # Load the cluster assignments and check their structure
            assignments = ClusterAssignmentLoader.load_cluster_assigment(
                "cluster_assignments.torch"
            )
            self.assertEqual(40, len(assignments.cluster_assignments["TRAIN"]))
            self.assertEqual(20, len(assignments.cluster_assignments["TEST"]))

    @gpu_test(gpu_count=2)
    def test_extract_cluster_assignment_ddp(self) -> None:
        self.run_cluster_assignment(with_fsdp=False)

    @gpu_test(gpu_count=2)
    def test_extract_cluster_assignment_fsdp(self) -> None:
        self.run_cluster_assignment(with_fsdp=True)
