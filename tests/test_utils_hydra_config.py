# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    SweepHydraOverrides,
)


class TestUtilsHydraConfig(unittest.TestCase):
    @staticmethod
    def _create_config(overrides: List[str]):
        cfg = compose_hydra_configuration(overrides)
        args, config = convert_to_attrdict(cfg, dump_config=False)
        return config

    def test_composition_order(self) -> None:
        """
        Following Hydra 1.1 update, the composition order is modified:
        https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order/

        This test verifies that the composition order is working on
        the hydra version installed by the user.
        """

        # Create a pre-training configuration for SwAV and
        # ensure that the composition was done correctly
        config = self._create_config(
            [
                "config=test/integration_test/quick_swav",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
            ]
        )
        self.assertEqual("swav_loss", config.LOSS.name)
        self.assertEqual(2, config.DISTRIBUTED.NUM_PROC_PER_NODE)

    def test_inference_of_fsdp_settings_for_swav_pretraining(self) -> None:
        overrides = [
            "config=pretrain/swav/swav_8node_resnet",
            "+config/pretrain/swav/models=regnet16Gf",
            "config.MODEL.AMP_PARAMS.USE_AMP=True",
            "config.OPTIMIZER.use_larc=True",
        ]

        cfg = self._create_config(overrides)
        self.assertEqual(cfg["MODEL"]["AMP_PARAMS"]["AMP_TYPE"], "apex")
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "swav_head")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_task")
        self.assertEqual(cfg.OPTIMIZER.name, "sgd")

        cfg = self._create_config(
            overrides + ["config.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP=True"]
        )
        self.assertEqual(cfg["MODEL"]["AMP_PARAMS"]["AMP_TYPE"], "pytorch")
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "swav_head_fsdp")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet_fsdp")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_fsdp_task")
        self.assertEqual(cfg.OPTIMIZER.name, "sgd_fsdp")

    def test_inference_of_fsdp_settings_for_linear_evaluation(self) -> None:
        overrides = [
            "config=debugging/benchmark/linear_image_classification/eval_resnet_8gpu_transfer_imagenette_160",
            "+config/debugging/benchmark/linear_image_classification/models=regnet16Gf_mlp",
        ]

        cfg = self._create_config(overrides)
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "mlp")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_task")

        cfg = self._create_config(
            overrides + ["config.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP=True"]
        )
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "mlp_fsdp")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet_fsdp")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_fsdp_task")

    def test_inference_of_fsdp_settings_for_linear_evaluation_with_bn(self) -> None:
        overrides = [
            "config=debugging/benchmark/linear_image_classification/eval_resnet_8gpu_transfer_imagenette_160",
            "+config/debugging/benchmark/linear_image_classification/models=regnet16Gf_eval_mlp",
        ]

        cfg = self._create_config(overrides)
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "eval_mlp")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_task")

        cfg = self._create_config(
            overrides + ["config.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP=True"]
        )
        self.assertEqual(cfg.MODEL.HEAD.PARAMS[0][0], "eval_mlp_fsdp")
        self.assertEqual(cfg.MODEL.TRUNK.NAME, "regnet_fsdp")
        self.assertEqual(cfg.TRAINER.TASK_NAME, "self_supervision_fsdp_task")

    def test_hyper_parameter_sweeps(self) -> None:
        config = "config=debugging/benchmark/linear_image_classification/eval_resnet_8gpu_transfer_imagenette_160"
        over_1 = "+config/debugging/benchmark/linear_image_classification/models=regnet16Gf_eval_mlp"
        sweep_1 = "config.OPTIMIZER.weight_decay=0.001,0.0001"
        sweep_2 = "config.OPTIMIZER.num_epochs=50,100,200"
        cli_overrides = [config, over_1, sweep_1, sweep_2]
        overrides, sweeps = SweepHydraOverrides.from_overrides(cli_overrides)
        self.assertEqual(overrides, [config, over_1])
        self.assertEqual(len(sweeps), 6)
        self.assertEqual(
            sweeps[0],
            ["config.OPTIMIZER.weight_decay=0.001", "config.OPTIMIZER.num_epochs=50"],
        )
        self.assertEqual(
            sweeps[1],
            ["config.OPTIMIZER.weight_decay=0.001", "config.OPTIMIZER.num_epochs=100"],
        )
        self.assertEqual(
            sweeps[2],
            ["config.OPTIMIZER.weight_decay=0.001", "config.OPTIMIZER.num_epochs=200"],
        )
        self.assertEqual(
            sweeps[3],
            ["config.OPTIMIZER.weight_decay=0.0001", "config.OPTIMIZER.num_epochs=50"],
        )
        self.assertEqual(
            sweeps[4],
            ["config.OPTIMIZER.weight_decay=0.0001", "config.OPTIMIZER.num_epochs=100"],
        )
        self.assertEqual(
            sweeps[5],
            ["config.OPTIMIZER.weight_decay=0.0001", "config.OPTIMIZER.num_epochs=200"],
        )
