# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum, auto
from typing import List

from classy_vision.hooks.classy_hook import ClassyHook
from vissl.config import AttrDict
from vissl.hooks.deepclusterv2_hooks import ClusterMemoryHook, InitMemoryHook  # noqa
from vissl.hooks.grad_clip_hooks import GradClipHook  # noqa
from vissl.hooks.log_hooks import (  # noqa
    LogGpuMemoryHook,
    LogGpuStatsHook,
    LogLossLrEtaHook,
    LogLossMetricsCheckpointHook,
    LogPerfTimeMetricsHook,
)
from vissl.hooks.moco_hooks import MoCoHook  # noqa
from vissl.hooks.profiling_hook import ProfilingHook
from vissl.hooks.state_update_hooks import (  # noqa
    CheckNanLossHook,
    FreezeParametersHook,
    SetDataSamplerEpochHook,
    SSLModelComplexityHook,
    UpdateBatchesSeenHook,
    UpdateTestBatchTimeHook,
    UpdateTrainBatchTimeHook,
    UpdateTrainIterationNumHook,
)
from vissl.hooks.swav_hooks import NormalizePrototypesHook  # noqa
from vissl.hooks.swav_hooks import SwAVUpdateQueueScoresHook  # noqa
from vissl.hooks.swav_momentum_hooks import (
    SwAVMomentumHook,
    SwAVMomentumNormalizePrototypesHook,
)
from vissl.hooks.tensorboard_hook import SSLTensorboardHook  # noqa
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.tensorboard import get_tensorboard_hook, is_tensorboard_available


class SSLClassyHookFunctions(Enum):
    """
    Enumeration of all the hook functions in the ClassyHook class.
    """

    on_start = auto()
    on_phase_start = auto()
    on_forward = auto()
    on_loss_and_meter = auto()
    on_backward = auto()
    on_update = auto()
    on_step = auto()
    on_phase_end = auto()
    on_end = auto()


def default_hook_generator(cfg: AttrDict, event_storage) -> List[ClassyHook]:
    """
    The utility function that prepares all the hoooks that will be used in training
    based on user selection. Some basic hooks are used by default.

    Optional hooks:
        - Tensorboard hook,
        - loss specific hooks (swav loss, deepcluster loss, moco loss) used only when the
          loss is being used
        - model complexity hook (if user wants to compute model flops, activations, params)
          enable the hook via HOOKS.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY = True

    Returns:
        hooks (List(functions)): list containing the hook functions that will be used
    """
    hooks = []

    # conditionally add hooks based on use-case
    if cfg.HOOKS.PERF_STATS.MONITOR_PERF_STATS:
        perf_stat_freq = (
            cfg.HOOKS.PERF_STATS.PERF_STAT_FREQUENCY
            if cfg.HOOKS.PERF_STATS.PERF_STAT_FREQUENCY > 0
            else None
        )
        hooks.append(LogPerfTimeMetricsHook(perf_stat_freq))
    if cfg.LOSS.name == "swav_loss":
        hooks.extend([SwAVUpdateQueueScoresHook(), NormalizePrototypesHook()])
    if cfg.LOSS.name == "swav_momentum_loss":
        hooks.extend(
            [
                SwAVMomentumHook(
                    cfg.LOSS["swav_momentum_loss"]["momentum"],
                    cfg.LOSS["swav_momentum_loss"]["momentum_eval_mode_iter_start"],
                    cfg.LOSS["swav_momentum_loss"]["crops_for_assign"],
                ),
                SwAVMomentumNormalizePrototypesHook(),
            ]
        )
    if cfg.LOSS.name == "deepclusterv2_loss":
        hooks.extend([InitMemoryHook(), ClusterMemoryHook()])
    if cfg.LOSS.name == "moco_loss":
        hooks.extend(
            [
                MoCoHook(
                    cfg.LOSS["moco_loss"]["momentum"],
                    shuffle_batch=(not cfg.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN),
                )
            ]
        )
    if cfg.HOOKS.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY:
        hooks.extend([SSLModelComplexityHook()])
    if cfg.HOOKS.LOG_GPU_STATS:
        hooks.extend([LogGpuStatsHook()])
    if cfg.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY:
        hooks.extend([LogGpuMemoryHook(cfg.HOOKS.MEMORY_SUMMARY.LOG_ITERATION_NUM)])
    if cfg.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD:
        assert is_tensorboard_available(), (
            "Tensorboard must be installed to use it. Please install tensorboard using:"
            "If pip environment: `pip install tensorboard` "
            "If using conda and you prefer conda install of tensorboard: "
            "`conda install -c conda-forge tensorboard`"
        )
        tb_hook = get_tensorboard_hook(cfg, event_storage)
        hooks.extend([tb_hook])
    if cfg.MODEL.GRAD_CLIP.USE_GRAD_CLIP:
        hooks.extend(
            [
                GradClipHook(
                    norm_type=cfg.MODEL.GRAD_CLIP.NORM_TYPE,
                    max_norm=cfg.MODEL.GRAD_CLIP.MAX_NORM,
                )
            ]
        )

    # hooks that are used irrespective of workflow type
    rolling_btime_freq = (
        cfg.HOOKS.PERF_STATS.ROLLING_BTIME_FREQ
        if cfg.HOOKS.PERF_STATS.ROLLING_BTIME_FREQ > 0
        else None
    )

    if ProfilingHook.is_enabled(cfg.PROFILING):
        hooks.append(ProfilingHook(profiling_config=cfg.PROFILING))

    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    checkpoint_folder = get_checkpoint_folder(cfg)
    hooks.extend(
        [
            CheckNanLossHook(),
            SetDataSamplerEpochHook(),
            FreezeParametersHook(),
            UpdateBatchesSeenHook(),
            UpdateTrainBatchTimeHook(),
            UpdateTestBatchTimeHook(),
            UpdateTrainIterationNumHook(),
            LogLossMetricsCheckpointHook(world_size),
            LogLossLrEtaHook(checkpoint_folder, rolling_btime_freq),
        ]
    )
    return hooks
