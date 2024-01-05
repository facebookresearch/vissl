# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from classy_vision.generic.util import load_checkpoint
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestEmaHook(unittest.TestCase):
    """
    Check that EMA Hook works correctly.
    Check that EMA model is saved in the checkpoint
    and that the accuracies are logged to the metrics.json
    """

    @gpu_test(gpu_count=1)
    def test_ema_hook(self) -> None:
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_in1k_linear.yaml",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.OPTIMIZER.num_epochs=2",
                "config.HOOKS.EMA_MODEL.SAVE_EMA_MODEL=True",
                "config.HOOKS.EMA_MODEL.ENABLE_EMA_METERS=True",
                "config.HOOKS.EMA_MODEL.EMA_DEVICE=gpu",
            ],
        )
        _, config = convert_to_attrdict(cfg)

        with in_temporary_directory() as checkpoint_folder:
            # Run a quick_eval_in1k_linear.
            integration_logs = run_integration_test(config)
            checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.torch")

            # Test that the ema model is saved in the checkpoint.
            checkpoint = load_checkpoint(checkpoint_path)
            self.assertTrue(
                "ema_model" in checkpoint["classy_state_dict"].keys(),
                msg="ema_model has not been saved to the checkpoint folder.",
            )

            # Test that train_accuracy_list_meter_ema have been logged to metrics.json.
            metrics = integration_logs.get_accuracies(from_metrics_file=True)
            self.assertTrue(
                "train_accuracy_list_meter_ema" in metrics[1],
                msg="train_accuracy_list_meter_ema is not logged to the metrics.json file.",
            )

            self.assertEqual(
                len(metrics),
                8,
                "the metrics.json output does not have the appropriate number of entries.",
            )
