# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum, auto
from typing import List

from classy_vision.hooks.classy_hook import ClassyHook
from vissl.hooks.deepclusterv2_hooks import ClusterMemoryHook, InitMemoryHook  # noqa
from vissl.hooks.log_hooks import (  # noqa
    LogGpuStatsHook,
    LogLossLrEtaHook,
    LogLossMetricsCheckpointHook,
    LogPerfTimeMetricsHook,
)
from vissl.hooks.moco_hooks import MoCoHook  # noqa
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
from vissl.utils.hydra_config import AttrDict
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


def default_hook_generator(cfg: AttrDict) -> List[ClassyHook]:
    """
    The utility function that prepares all the hoooks that will be used in training
    based on user selection. Some basic hooks are used by default.

    Optional hooks:
        - Tensorboard hook,
        - loss specific hooks (swav loss, deepcluster loss, moco loss) used only when the
          loss is being used
        - model complexity hook (if user wants to compute model flops, activations, params)
          enable the hook via MODEL.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY = True

    Returns:
        hooks (List(functions)): list containing the hook functions that will be used
    """
    hooks = []

    # conditionally add hooks based on use-case
    if cfg.MONITOR_PERF_STATS:
        perf_stat_freq = (
            cfg.PERF_STAT_FREQUENCY if cfg.PERF_STAT_FREQUENCY > 0 else None
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
    if cfg.MODEL.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY:
        hooks.extend([SSLModelComplexityHook()])
    if cfg.TENSORBOARD_SETUP.USE_TENSORBOARD:
        assert is_tensorboard_available(), "Tensorboard must be installed to use it."
        tb_hook = get_tensorboard_hook(cfg)
        hooks.extend([tb_hook])

    # hooks that are used irrespective of workflow type
    rolling_btime_freq = cfg.ROLLING_BTIME_FREQ if cfg.ROLLING_BTIME_FREQ > 0 else None
    hooks.extend(
        [
            CheckNanLossHook(),
            SetDataSamplerEpochHook(),
            FreezeParametersHook(),
            UpdateBatchesSeenHook(),
            UpdateTrainBatchTimeHook(),
            UpdateTestBatchTimeHook(),
            UpdateTrainIterationNumHook(),
            LogLossMetricsCheckpointHook(),
            LogLossLrEtaHook(rolling_btime_freq),
            LogGpuStatsHook(),
        ]
    )
    return hooks
