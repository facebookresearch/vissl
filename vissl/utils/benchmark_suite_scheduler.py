# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import submitit
from iopath.common.file_io import g_pathmgr
from vissl.config.attr_dict import AttrDict
from vissl.utils.distributed_launcher import launch_distributed_on_slurm
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.io import load_file, makedir
from vissl.utils.misc import flatten_dict, retry


"""
This class is designed to be used to run multiple evaluations on a single (pre)training.
Using the #evaluate method we continuously monitor training checkpoints, launch evaluations
dynamically as they become available, and amalgamate the evaluation results as they become
available.

For SLURM usage, you should create a JSON configuration file
(see benchmark_suite_scheduler_template.json) and use
launch_benchmark_suite_scheduler_slurm.sh for convenience.
"""

_DEFAULT_PYTORCH_PORTS = [40050]
# How many times to retry a slurm job submission.
_NUM_SLURM_RETRIES = 5
# How many seconds to sleep between iterations of the main loop.
_SLEEP_TIME_SECONDS = 15
# Slurm states marked as terminal. SlurmEvulator#evaluate will finish
# once all jobs are in a terminal state.
_SLURM_JOB_TERMINAL_STATES = [
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "SUSPENDED",
    "TIMEOUT",
]
# Wait for the training checkpoint folder to be available for 1 hour.
_TRAINING_CONFIG_WAIT_SECONDS = 60 * 60


class BenchmarkSuiteScheduler:
    """
    The Slurm Evaluator is a class designed to continuously monitor VISSL pretrainings
    and launch evaluations as checkpoints become available. The method takes a
    config dictionary consisting of the training checkpoint directory, an array of
    benchmarks, and information on how often to evaluate the trainings.
    """

    def __init__(
        self,
        training_checkpoint_dir: str,
        benchmarks: List,
        evaluate_final_phase=True,
        evaluation_phase_freq: int = -1,
        evaluation_iter_freq: int = -1,
        autoload_slurm_evaluator_checkpoint=False,
        slurm_evaluator_checkpoint: str = None,
        retry_evaluation_job_ids: List[int] = None,
        auto_retry_evaluations=False,
        max_retries=3,
        pytorch_ports=None,
    ):
        """
        Args:

        training_checkpoint_dir: (str). Checkpoint directory of the training.
                                This should match the trainings CHECKPOINT.dir
        benchmarks: (list[dict]) Benchmarks with the following structure:
            "config_files": [
                    {
                        # Path to config file.
                        "config=test/integration_test/quick_eval_in1k_linear.yaml"
                        # Config overrides.
                        "config.TRAIN.DATA_LIMIT=1000",
                        ...
                    },
                    ...
                ]
        evaluate_final_phase: (bool, optional, default=True). Whether or not to evaluate the
                                final phase of the training.
        evaluation_phase_freq: (int, optional, default=-1) How often to evaluate phases.
                                Training checkpoint phase freq must evenly
                                divide evaluation_phase_freq.
        evaluation_iter_freq: (int, optional, default=-1) How often to evaluate iterations.
                               Training checkpoint iteration freq must evenly
                               divide evaluation_iter_freq.
        autoload_slurm_evaluator_checkpoint: (bool, optional, default=False) Whether or not to
                                              autoload slurm_evaluator Checkpoint.
                                              This is useful when slurm evaluator job
                                              is preempted for example.
        slurm_evaluator_checkpoint: (str, optional, default=None) String of
                                    slurm_evaluator checkpoint directory.
        retry_evaluation_job_ids: (List[int], optional, default=[]) List of job_ids to retry.
        auto_retry_evaluations: (bool, optional, default=False) Whether or not to automatically
                                retry all failed jobs.
        max_retries: (int, optional, default=3). Maximum number of retries.
        pytorch_ports: (list[int], optional, default=[40050]). Ports to cycle through as
                        you are launching your trainings.
        """
        self.evaluation_jobs_finished = set()

        # Required Arguments
        self.training_checkpoint_dir = training_checkpoint_dir
        self.training_checkpoint_file = os.path.join(
            self.training_checkpoint_dir, "train_config.yaml"
        )
        self.benchmarks = benchmarks

        # Optional Arguments
        self.evaluate_final_phase = evaluate_final_phase
        self.evaluation_phase_freq = evaluation_phase_freq
        self.evaluation_iter_freq = evaluation_iter_freq
        self.autoload_slurm_evaluator_checkpoint = autoload_slurm_evaluator_checkpoint
        self.slurm_evaluator_checkpoint = slurm_evaluator_checkpoint
        self.retry_evaluation_job_ids = retry_evaluation_job_ids or []
        self.auto_retry_evaluations = auto_retry_evaluations
        self.max_retries = max_retries
        self.pytorch_ports = pytorch_ports or _DEFAULT_PYTORCH_PORTS
        self.pytorch_ports_iterable = iter(self.pytorch_ports)

        self.validate()

        # Will be set in #evaluate, once training_checkpoint_dir becomes available.
        self.training_config = None
        self.evaluation_results = None

    def evaluate(self):
        """
        Evaluate the checkpoints. At a high level, this is the structure.

        1. Load training YAML config file.
        2. Monitor training checkpoints. When checkpoint is ready, launch the evaluation jobs.
        3. Monitor the evaluation jobs. When evaluation jobs are complete, mark the results.
        3. Once all jobs have been recorded, complete.
        """
        start_time = time.time()

        # Wait for the training config to be available. This indicates the training has begun.
        while True:
            if time.time() - start_time > _TRAINING_CONFIG_WAIT_SECONDS:
                raise RuntimeError(
                    f"Training config still doesn't exist after:"
                    f"{_TRAINING_CONFIG_WAIT_SECONDS / 60} minutes"
                )

            if (
                g_pathmgr.exists(self.training_checkpoint_file)
                and self._max_training_iterations()
            ):
                # Load training yaml config.
                self._load_training_config()

                # Set max training iterations
                self.max_training_iterations = self._max_training_iterations()

                # Generate evaluation results
                self.evaluation_results = self._generate_initial_benchmark_results()
                self._validate_evaluation_setup()

                break

            time.sleep(_SLEEP_TIME_SECONDS)

        # Save initial evaluation benchmarks, for checkpointing reasons.
        self.save_evaluation_benchmarks()

        # Checkpoint folder is now available. Continuously monitor the training checkpoints,
        # launch evaluation jobs as needed, monitor their progress, and record their results.
        while True:
            self._evaluate_checkpoints()

            self._check_evaluation_jobs()

            # Break if no more checkpoints to evaluate
            if self._finished():
                logging.info("Evaluations are finished")
                break

            time.sleep(_SLEEP_TIME_SECONDS)

    def _max_training_iterations(self):
        """
        Get the max number of training iterations for the main SSL training.
        """
        training_stdout_json_file = os.path.join(
            self.training_checkpoint_dir, "stdout.json"
        )

        # If the stdout.json path doesn't exist, return None.
        if not g_pathmgr.exists(training_stdout_json_file):
            return None

        with g_pathmgr.open(training_stdout_json_file, "rb") as f:
            # First line of stdout.json must have max_iterations in the first line
            try:
                first_json_line = json.loads(next(f))
                assert (
                    "max_iterations" in first_json_line
                ), "Training must set max_iterations in the stoud.json. See LogLossLrEtaHook."
                return first_json_line["max_iterations"]
            except StopIteration:
                return None

    def save_evaluation_benchmarks(self):
        """
        Create the /evaluations directory inside the training checkpoints dir.
        Upload json file to the parent evaluation directories, as well as
        to each child evaluation directories.
        """
        # Upload all checkpoints evaluations to parent checkpoint directory.
        evaluation_dir = self.evaluation_dir()
        parent_metrics_file = os.path.join(evaluation_dir, "evaluation_metrics.json")

        makedir(evaluation_dir)

        self._write_json_file(self.evaluation_results, parent_metrics_file)

        # Upload each checkpoint's evaluations to child directories.
        for checkpoint_str, benchmarks in self.evaluation_results.items():
            child_metrics_dir = os.path.join(evaluation_dir, checkpoint_str)
            child_metrics_file = os.path.join(
                child_metrics_dir, "evaluation_metrics.json"
            )

            makedir(child_metrics_dir)

            self._write_json_file(benchmarks, child_metrics_file)

        logging.info("Saved benchmarks json file.")

    def evaluation_dir(self):
        return os.path.join(self.training_checkpoint_dir, "evaluations")

    def _load_training_config(self):
        # Load training yaml config.
        self.training_config = load_file(self.training_checkpoint_file)
        self.training_config = AttrDict(self.training_config)

        logging.info(
            f"Loaded training checkpoint config from: { self.training_checkpoint_file }"
        )

    def validate(self):
        """
        Validate the class instance is valid.
        """
        assert not (
            self.autoload_slurm_evaluator_checkpoint and self.slurm_evaluator_checkpoint
        ), "Specify only one of autoload_slurm_evaluator_checkpoint and slurm_evaluator_checkpoint"  # NOQA
        assert (
            type(self.evaluation_iter_freq) is int and self.evaluation_iter_freq >= -1
        ), "The evaluation_iter_freq must be an int >= 1"
        assert (
            type(self.evaluation_phase_freq) is int and self.evaluation_phase_freq >= -1
        ), "The evaluation_phase_freq must be an int >= 1"
        assert (
            self.evaluation_iter_freq >= -1
            or self.evaluation_phase_freq >= -1
            or self.evaluate_final_phase
        ), "Please specify evaluation_iter_freq, evaluation_phase_freq, or evaluate_final_phase"  # NOQA
        assert (
            type(self.max_retries) is int and self.max_retries >= -1
        ), "Max retries must be >= -1."

    def _validate_evaluation_setup(self):
        if self.evaluation_iter_freq > -1:
            assert (
                self.evaluation_iter_freq
                % self.training_config.CHECKPOINT.CHECKPOINT_ITER_FREQUENCY
            ) == 0, "Evaluation iter frequency must evenly divide the checkpoint iter frequency"  # NOQA

        if self.evaluation_phase_freq > -1:
            assert (
                self.evaluation_phase_freq
                % self.training_config.CHECKPOINT.CHECKPOINT_FREQUENCY
            ) == 0, "Evaluation phase frequency must evenly divide the checkpoint phase frequency"  # NOQA

        assert g_pathmgr.exists(
            self.training_config.SLURM.LOG_FOLDER
        ), "Training slurm log folder must exist"
        assert g_pathmgr.exists(
            self.training_config.CHECKPOINT.DIR
        ), "Training slurm checkpoint folder must exist"

    def _finished(self):
        # Count total number of evaluation jobs.
        total_jobs = 0
        for benchmarks in self.evaluation_results.values():
            total_jobs += len(benchmarks)

        return len(self.evaluation_jobs_finished) == total_jobs

    def _evaluate_checkpoints(self):
        for checkpoint_str, benchmarks in self.evaluation_results.items():
            # TODO: Can we possible retrieve this from CheckpointWriter, to consolidate logic.
            checkpoint_str = os.path.join(
                self.training_config.CHECKPOINT.DIR, f"{ checkpoint_str }.torch"
            )
            if g_pathmgr.exists(checkpoint_str):
                self._evaluate_checkpoint(checkpoint_str, benchmarks)

    def _evaluate_checkpoint(self, checkpoint_str, benchmarks):
        for benchmark in benchmarks:
            retry_job = self._retry_job(benchmark)
            if benchmark["job_id"] and not retry_job:
                continue

            if retry_job:
                self.evaluation_jobs_finished.remove(benchmark["job_id"])
                # Log the job retry.
                job_id, slurm_state = benchmark["job_id"], benchmark["slurm_state"]
                logging.info(f"Retrying job: { job_id } in state: { slurm_state }")

            args, config = self._generate_config(benchmark["config_files"])
            job = self._launch_slurm_job(args, config)

            time.sleep(10)  # Wait for slurm job status to be reliably updated.

            # Set checkpoint status
            benchmark["job_id"] = job.job_id
            benchmark["num_retries"] += 1
            benchmark["slurm_log_dir"] = config.SLURM.LOG_FOLDER
            benchmark["slurm_checkpoint_dir"] = config.CHECKPOINT.DIR
            benchmark["weights_init_params_file"] = (
                config.MODEL.WEIGHTS_INIT.PARAMS_FILE
            )
            benchmark["slurm_state"] = job.state

            current_time = datetime.now().strftime("%H:%M:%S %z")
            log = f"""
                Launched Slurm Evaluation job. Time: { current_time }
                job_id: { job.job_id }, num_retries: { benchmark["num_retries"] }
                evaluation_name: { benchmark["evaluation_name"] }
                checkpoint_str: { checkpoint_str }
                state_prev: None, state_current: { job.state }
            """

            logging.info(log)

            # Save evaluation results to json file.
            self.save_evaluation_benchmarks()

    def _retry_job(self, benchmark):
        return benchmark["job_id"] in self.retry_evaluation_job_ids or (
            benchmark["slurm_state"] in _SLURM_JOB_TERMINAL_STATES
            and benchmark["slurm_state"] != "COMPLETED"
            and self.auto_retry_evaluations
            and benchmark["num_retries"] < self.max_retries
        )

    @retry(n_tries=_NUM_SLURM_RETRIES)
    def _launch_slurm_job(self, args, config):
        # Get next port in the list of #pytorch_ports
        try:
            port = next(self.pytorch_ports_iterable)
        except StopIteration:
            # Start at the beginning of the ports list.
            self.pytorch_ports_iterable = iter(self.pytorch_ports)
            port = next(self.pytorch_ports_iterable)

        config.SLURM.PORT_ID = port

        return launch_distributed_on_slurm(engine_name=args.engine_name, cfg=config)

    def _write_json_file(self, data, file_name):
        with g_pathmgr.open(file_name, "w") as fopen:
            fopen.write(json.dumps(data, sort_keys=True))
            fopen.flush()

    def _check_evaluation_jobs(self):
        # Monitor each evaluation job, change slurm job state as needed, and
        # load results if finished.
        for benchmarks in self.evaluation_results.values():
            for benchmark in benchmarks:
                self._monitor_benchmark_job(benchmark)

    def _monitor_benchmark_job(self, benchmark):
        if not benchmark["job_id"]:
            return  # Do nothing, the job has not yet started.

        # Create SlurmJob object.
        job_id = str(benchmark["job_id"])
        folder = Path(benchmark["slurm_log_dir"])
        job = submitit.SlurmJob(job_id=job_id, folder=folder, tasks=[0])

        if job.state in _SLURM_JOB_TERMINAL_STATES:
            # Job is in terminal state, mark job as finished.
            self.evaluation_jobs_finished.add(job.job_id)

        if job.state != benchmark["slurm_state"]:
            # Job state has changed, log transition, and update state in json file.
            checkpoint_str = os.path.split(benchmark["weights_init_params_file"])[-1]

            current_time = datetime.now().strftime("%H:%M:%S %z")
            log = f"""
                Slurm Evaluation job changed states. Time: { current_time }
                job_id: { job.job_id }, num_retries: { benchmark["num_retries"] }
                evaluation_name: { benchmark["evaluation_name"] },
                checkpoint_str: { checkpoint_str },
                state_prev: { benchmark["slurm_state"] }, state_curr: { job.state }
            """

            logging.info(log)
            # Benchmark Job state has changed. Update the benchmark state.
            self._update_benchmark_state(benchmark, job)
            self.save_evaluation_benchmarks()

    def _update_benchmark_state(self, benchmark, job):
        # Job state has changed, record it.
        benchmark["slurm_state"] = job.state

        if job.done():
            # Upload metrics files.
            benchmark["metrics"] = self._get_benchmark_metrics(benchmark)

    def _get_benchmark_metrics(self, benchmark):
        metrics_file = os.path.join(benchmark["slurm_checkpoint_dir"], "metrics.json")

        if g_pathmgr.exists(metrics_file):
            # Open metrics file from finished evaluation job.
            metrics = []
            with g_pathmgr.open(metrics_file, "rb") as f:
                for line in f:
                    metrics.append(json.loads(line))

            final_metrics = collections.defaultdict(lambda: {"metric": -1})

            self._set_largest_metric(metrics, final_metrics)

            result = dict(final_metrics)
        else:
            result = """Evaluation Job has completed, but metrics.json is not available.
                        Please check the evaluation's checkpoint_dir."""

        return result

    def _set_largest_metric(self, metrics, final_metrics):
        # Get the largest metrics over all recorded metrics.
        for m in metrics:
            flattened_metrics = flatten_dict(m)
            for metric_name, metric in flattened_metrics.items():
                if metric_name in ["iteration", "phase_idx", "train_phase_idx"]:
                    continue  # These are not evaluation metrics

                if metric > final_metrics[metric_name]["metric"]:
                    final_metrics[metric_name]["metric"] = metric
                    final_metrics[metric_name]["iteration"] = flattened_metrics[
                        "iteration"
                    ]
                    final_metrics[metric_name]["train_phase_idx"] = flattened_metrics[
                        "train_phase_idx"
                    ]

    def _generate_initial_benchmark_results(self):
        default_checkpoint = os.path.join(
            self.evaluation_dir(), "evaluation_metrics.json"
        )
        autoload_slurm_evaluator_checkpoint = (
            self.autoload_slurm_evaluator_checkpoint
            and g_pathmgr.exists(default_checkpoint)
        )

        if autoload_slurm_evaluator_checkpoint or self.slurm_evaluator_checkpoint:
            return self._load_evaluation_results_checkpoint()

        evaluation_configs = {}

        for benchmark in self.benchmarks:
            default_evaluation_name = os.path.split(benchmark["config_files"][0])[-1]
            evaluation_name = (
                benchmark.get("evaluation_name") or default_evaluation_name
            )

            last_phase = self.training_config.OPTIMIZER.num_epochs - 1

            # TODO: Can we retrieve this from CheckpointWriter?
            if self.evaluate_final_phase:
                # Evaluate Last phase checkpoint
                training_checkpoint = f"model_final_checkpoint_phase{ last_phase }"
                self._set_initial_benchmark_result(
                    benchmark, training_checkpoint, evaluation_name, evaluation_configs
                )

            if self.evaluation_phase_freq > -1:
                # Evaluate every "evaluation_phase_freq" phase checkpoint.
                evaluate_epochs = range(self.evaluation_phase_freq, last_phase)[
                    :: self.evaluation_phase_freq
                ]
                for epoch in evaluate_epochs:
                    training_checkpoint = f"model_phase{epoch}"
                    self._set_initial_benchmark_result(
                        benchmark,
                        training_checkpoint,
                        evaluation_name,
                        evaluation_configs,
                    )

            if self.evaluation_iter_freq > -1:
                # Evaluate every "evaluation_iter_freq" iteration checkpoints.
                evaluate_iterations = range(
                    self.evaluation_iter_freq, self.max_training_iterations
                )[:: self.evaluation_iter_freq]
                for iteration in evaluate_iterations:
                    training_checkpoint = f"model_iteration{iteration}"
                    self._set_initial_benchmark_result(
                        benchmark,
                        training_checkpoint,
                        evaluation_name,
                        evaluation_configs,
                    )

        return evaluation_configs

    def _load_evaluation_results_checkpoint(self):
        default_checkpoint = os.path.join(
            self.evaluation_dir(), "evaluation_metrics.json"
        )
        checkpoint_file = (
            default_checkpoint
            if self.autoload_slurm_evaluator_checkpoint
            else self.slurm_evaluator_checkpoint
        )

        evaluation_config = load_file(checkpoint_file)

        logging.info(f"Loaded evaluation results checkpoint from: { checkpoint_file }")

        return evaluation_config

    def _set_initial_benchmark_result(
        self, benchmark, training_checkpoint, evaluation_name, evaluation_configs
    ):
        """
        Generates evaluation configs in order to evaluate the final output of the
        pretraining model specified in 'training_config'.
        """
        log_dir = self._evaluation_log_dir(training_checkpoint, evaluation_name)
        checkpoint_dir = self._evaluation_checkpoint_dir(
            training_checkpoint, evaluation_name
        )

        evaluation_configs[training_checkpoint] = (
            evaluation_configs.get(training_checkpoint) or []
        )

        # Add benchmark result information
        benchmark_result = {
            "evaluation_name": evaluation_name,
            "job_id": None,
            "num_retries": 0,
            "slurm_log_dir": log_dir,
            "checkpoint_dir": checkpoint_dir,
            "metrics": None,
            "slurm_state": None,
            "config_files": benchmark["config_files"].copy(),
        }

        # Hydra config information
        weights_init_path = os.path.join(
            self.training_config.CHECKPOINT.DIR, f"{ training_checkpoint }.torch"
        )

        # Override certain options for slurm
        for option in [
            f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE='{weights_init_path}'",
            "config.SLURM.USE_SLURM=true",
            f"config.SLURM.LOG_FOLDER='{log_dir}'",
            f"config.CHECKPOINT.DIR='{checkpoint_dir}'",
            f"hydra.run.dir='{ log_dir }'",
        ]:
            benchmark_result["config_files"].insert(1, option)

        evaluation_configs[training_checkpoint].append(benchmark_result)

    def _evaluation_log_dir(self, evaluation_directory, evaluation_name):
        """
        Directory to put logs for an evaluation job.
        """
        evaluation_dir = self.evaluation_dir()
        return os.path.join(evaluation_dir, evaluation_directory, evaluation_name)

    def _evaluation_checkpoint_dir(self, model_final_checkpoint, evaluation_name):
        """
        Directory to put checkpoints in for an evaluation job.
        """
        return os.path.join(
            self._evaluation_log_dir(model_final_checkpoint, evaluation_name),
            "checkpoints",
        )

    def _generate_config(self, overrides: List[str]):
        """
        Generate AttrDict config from a config YAML file and overrides.
        """
        cfg = compose_hydra_configuration(overrides)
        return convert_to_attrdict(cfg)
