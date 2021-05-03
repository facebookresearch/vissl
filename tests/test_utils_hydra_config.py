# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize_config_module, compose
from vissl.utils.hydra_config import convert_to_attrdict


class TestUtilsHydraConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module(config_module="vissl.config")

    @staticmethod
    def _create_config(overrides: List[str]):
        cfg = compose("defaults", overrides=overrides)
        args, config = convert_to_attrdict(cfg)
        return config

    def test_inference_of_fsdp_settings_for_swav_pretraining(self):
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

    def test_inference_of_fsdp_settings_for_linear_evaluation(self):
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

    def test_inference_of_fsdp_settings_for_linear_evaluation_with_bn(self):
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
