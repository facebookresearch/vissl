# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestProfiler(unittest.TestCase):
    """
    Test suite to check that the profiling tools are working properly
    during training and benchmark evaluation
    """

    @staticmethod
    def _create_config(force_legacy_profiler: bool):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_simclr",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.OPTIMIZER.use_larc=False",
                "config.PROFILING.RUNTIME_PROFILING.USE_PROFILER=true",
                "config.PROFILING.PROFILED_RANKS=[0]",
                f"config.PROFILING.RUNTIME_PROFILING.LEGACY_PROFILER={force_legacy_profiler}",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=1)
    def test_profiler(self) -> None:
        with in_temporary_directory() as output_dir:
            config = self._create_config(force_legacy_profiler=False)
            run_integration_test(config)
            files = set(os.listdir(output_dir))
            print(files)
            self.assertIn("cuda_time_rank0.txt", files)
            self.assertIn("cuda_memory_usage_rank0.txt", files)
            self.assertIn("cpu_time_rank0.txt", files)
            self.assertIn("profiler_chrome_trace_rank0.json", files)

    @gpu_test(gpu_count=1)
    def test_legacy_profiler(self) -> None:
        with in_temporary_directory() as output_dir:
            config = self._create_config(force_legacy_profiler=True)
            run_integration_test(config)
            files = set(os.listdir(output_dir))
            print(files)
            self.assertIn("cuda_time_rank0.txt", files)
            self.assertIn("cuda_memory_usage_rank0.txt", files)
            self.assertIn("cpu_time_rank0.txt", files)
            self.assertIn("profiler_chrome_trace_rank0.json", files)
