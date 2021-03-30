import unittest

import torch
import torchvision.models as models
from vissl.utils.activation_statistics import (
    ActivationStatisticsAccumulator,
    ActivationStatisticsMonitor,
)


class TestDataLimitSubSampling(unittest.TestCase):
    def test_activation_statistics(self):
        torch.manual_seed(0)

        accumulator = ActivationStatisticsAccumulator()
        watcher = ActivationStatisticsMonitor(
            observer=accumulator, log_frequency=1, ignored_modules=set()
        )
        watcher.set_iteration(1)
        model = models.resnet18()
        watcher.monitor(model)
        model(torch.randn(size=(1, 3, 224, 224)))

        stats = accumulator.stats
        self.assertEqual(60, len(stats))
        for stat in stats:
            self.assertEqual(1, stat.iteration)

        # Verify that the first statistics produced is for the first
        # layer of the ResNet
        first_stat = stats[0]
        self.assertEqual("conv1", first_stat.name)
        self.assertEqual("torch.nn.modules.conv.Conv2d", first_stat.module_type)
        self.assertAlmostEqual(-0.0001686, first_stat.mean, delta=1e-6)
        self.assertAlmostEqual(1.6035640, first_stat.maxi, delta=1e-6)
        self.assertAlmostEqual(-1.4944878, first_stat.mini, delta=1e-6)

        # Verify that only leaf modules have statistics
        exported_modules_types = {
            "torch.nn.modules.pooling.MaxPool2d",
            "torch.nn.modules.pooling.AdaptiveAvgPool2d",
            "torch.nn.modules.batchnorm.BatchNorm2d",
            "torch.nn.modules.activation.ReLU",
            "torch.nn.modules.conv.Conv2d",
            "torch.nn.modules.linear.Linear",
        }
        module_types = {stat.module_type for stat in stats}
        self.assertEqual(sorted(exported_modules_types), sorted(module_types))
