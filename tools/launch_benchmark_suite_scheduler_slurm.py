# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import pkg_resources
import submitit
from iopath.common.file_io import g_pathmgr
from vissl.config.attr_dict import AttrDict
from vissl.utils.benchmark_suite_scheduler import BenchmarkSuiteScheduler
from vissl.utils.hydra_config import assert_hydra_dependency
from vissl.utils.io import load_file
from vissl.utils.misc import recursive_dict_merge
from vissl.utils.slurm import is_submitit_available


# Default config options
default_config_file = pkg_resources.resource_filename(
    "dev", "benchmark_suite/benchmark_suite_scheduler_defaults.json"
)
_DEFAULT_CONFIG = load_file(default_config_file)


class SlurmEvaluatorJob:
    """
    The slurm evaluator job is a thin wrapper around BenchmarkSuiteScheduler
    used by submitit. It's main function is to run multiple evaluations
    on a single training.
    """

    def __init__(self, benchmark_suite_scheduler: BenchmarkSuiteScheduler):
        self.benchmark_suite_scheduler = benchmark_suite_scheduler

    def __call__(self):
        self.benchmark_suite_scheduler.evaluate()

    def checkpoint(self):
        """
        This method is called whenever a job is pre-empted, timedout, etc,.
        Here we save the evaluation benchmarks, so that we can reload them
        and continue where we left off.
        """
        self.benchmark_suite_scheduler.save_evaluation_benchmarks()
        # Forces the benchmark_suite_scheduler to automatically reload it's
        # checkpoint, the benchmark results.
        self.benchmark_suite_scheduler.autoload_benchmark_suite_scheduler_checkpoint = (
            True
        )

        trainer = SlurmEvaluatorJob(
            benchmark_suite_scheduler=self.benchmark_suite_scheduler
        )
        return submitit.helpers.DelayedSubmission(trainer)


def launch_benchmark_suite_scheduler(config_file):
    assert g_pathmgr.exists(config_file), "Slurm evaluator config file must exist"

    user_config = load_file(config_file)
    config = _DEFAULT_CONFIG.copy()
    recursive_dict_merge(config, user_config)

    benchmark_suite_scheduler = BenchmarkSuiteScheduler(**config["params"])
    benchmark_suite_scheduler_job = SlurmEvaluatorJob(
        benchmark_suite_scheduler=benchmark_suite_scheduler
    )
    executor = submitit.AutoExecutor(folder=benchmark_suite_scheduler.evaluation_dir())

    assert "slurm_options" in config, "slurm_options must be specified"
    assert (
        "PARTITION" in config["slurm_options"]
    ), "slurm_options.PARTITION is a required field to launch the benchmark suite on slurm"

    slurm_options = AttrDict(config["slurm_options"])
    executor.update_parameters(
        name=slurm_options.NAME,
        slurm_comment=slurm_options.COMMENT,
        slurm_partition=slurm_options.PARTITION,
        slurm_constraint=slurm_options.CONSTRAINT,
        timeout_min=slurm_options.TIMEOUT_MIN,
        nodes=1,
        cpus_per_task=slurm_options.CPUS_PER_TASK,
        tasks_per_node=1,
        mem_gb=slurm_options.MEM_GB,
        slurm_additional_parameters=slurm_options.ADDITIONAL_PARAMETERS,
    )

    job = executor.submit(benchmark_suite_scheduler_job)
    print(f"SUBMITTED EVALUATION JOB: {job.job_id}")


if __name__ == "__main__":
    """
    Example usage:
    python -u "./vissl/engines/benchmark_suite_scheduler.py" \
        "/path/to/benchmark_suite_scheduler_example.json"
    """
    assert_hydra_dependency()

    assert (
        is_submitit_available()
    ), "Please 'pip install submitit' to schedule jobs on SLURM"

    config_file = sys.argv[1]
    launch_benchmark_suite_scheduler(config_file)
