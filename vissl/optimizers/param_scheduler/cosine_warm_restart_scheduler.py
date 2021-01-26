# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import math
from enum import Enum
from typing import Any, Dict

import numpy as np
from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    UpdateInterval,
    register_param_scheduler,
)


class CosineWaveTypes(str, Enum):
    half = "half"
    full = "full"


@register_param_scheduler("cosine_warm_restart")
class CosineWarmRestartScheduler(ClassyParamScheduler):
    """
    Changes the param value after every epoch based on a `cosine schedule <https:
    //arxiv.org/pdf/1608.03983.pdf>`_. The schedule is updated after every train
    step by default.

    Can be used for cosine learning rate with warm restarts. For restarts, we calculate
    what will be the maximum learning rate after every restart. There are 3 options:
        - Option 1: LR after every restart is same as original max LR
        - Option 2: LR after every restart decays with a fixed LR multiplier
        - Option 3: LR after every restart is adaptively calculated such that the resulting
                    max LR matches the original cosine wave LR.

    Args:
        wave_type: half | full
        lr_multiplier: float value -> LR after every restart decays with a fixed LR multiplier
        is_adaptive: True -> if after every restart, maximum LR is adaptively calculated such
                    that the resulting max LR matches the original cosine wave LR.
        update_interval: step | epoch -> if the LR should be updated after every training
                         iteration or after training epoch

    Example:

        .. code-block:: python

          start_value: 0.1
          end_value: 0.0001
          restart_interval_length: 0.5  # for 1 restart
          wave_type: half
          lr_multiplier: 1.0  # for using a decayed max LR value at every restart
          use_adaptive_decay: False
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        restart_interval_length: float,
        wave_type: str,
        lr_multiplier: float,
        is_adaptive: bool,
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        super().__init__(update_interval=update_interval)
        self.restart_interval_length = restart_interval_length
        self.wave_type = CosineWaveTypes(wave_type)
        self._start_value = start_value
        self._end_value = end_value
        self._max_lr = self._start_value
        self.lr_multiplier = lr_multiplier
        self.is_adaptive = is_adaptive

        self.restart_steps = []
        self.lr_restart_values = []
        self._init_restart_step_values()
        # we calculate what will be the maximum learning rate after every restart.
        # There are 3 options:
        # Option 1: LR after every restart is same as original max LR
        # Option 2: LR after every restart decays with a fixed LR multiplier
        # Option 3: LR after every restart is adaptively calculated such that the resulting
        #           max LR matches the original cosine wave LR.
        self._compute_lr_restart_values()

    def _init_restart_step_values(self):
        # calculate where the LR is restarted. We need this so we can find
        # what restart number we are at based on "where" we are in the training.
        self.restart_steps = [0.0]
        if self.wave_type == CosineWaveTypes.full:
            self.restart_steps.extend(
                np.arange(
                    self.restart_interval_length, 1.0, 2 * self.restart_interval_length
                ).tolist()
            )
        elif self.wave_type == CosineWaveTypes.half:
            self.restart_steps.extend(
                np.arange(
                    self.restart_interval_length, 1.0, self.restart_interval_length
                ).tolist()
            )

    def _compute_lr_restart_values(self):
        for idx in range(len(self.restart_steps)):
            if self.is_adaptive:
                if self.wave_type == CosineWaveTypes.half:
                    desired_max_lr = self._end_value + 0.5 * (
                        self._start_value - self._end_value
                    ) * (1 + math.cos(math.pi * self.restart_steps[idx]))
                elif self.wave_type == CosineWaveTypes.full:
                    desired_max_lr = self._end_value + 0.5 * (
                        self._start_value - self._end_value
                    ) * (
                        1
                        + math.cos(
                            math.pi
                            * (self.restart_interval_length + self.restart_steps[idx])
                        )
                    )
                lr_multiplier = desired_max_lr / self._start_value
            else:
                lr_multiplier = pow(self.lr_multiplier, idx)
            restart_lr = self._start_value * lr_multiplier
            self.lr_restart_values.append(restart_lr)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CosineWarmRestartScheduler":
        """
        Instantiates a CosineWarmRestartScheduler from a configuration.

        Args:
            config: A configuration for a CosineWarmRestartScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A CosineWarmRestartScheduler instance.
        """
        assert (
            "start_value" in config and "end_value" in config
        ), "CosineWarmRestartScheduler requires a start_value and a end_value"
        assert (
            "restart_interval_length" in config and "wave_type" in config
        ), "CosineWarmRestartScheduler requires restart_interval_length and wave_type"

        return cls(
            start_value=config["start_value"],
            end_value=config["end_value"],
            restart_interval_length=config["restart_interval_length"],
            wave_type=config.get("wave_type", CosineWaveTypes.half),
            lr_multiplier=config.get("lr_multiplier", 1.0),
            is_adaptive=config.get("is_adaptive", False),
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
        )

    def __call__(self, where: float):
        if self.wave_type == CosineWaveTypes.half:
            restart_num = max(bisect.bisect(self.restart_steps, where) - 1, 0)
            self._max_lr = self.lr_restart_values[restart_num]
            where = (where % float(self.restart_interval_length)) / float(
                self.restart_interval_length
            )
        elif self.wave_type == CosineWaveTypes.full:
            restart_num = max(bisect.bisect(self.restart_steps, where) - 1, 0)
            self._max_lr = self.lr_restart_values[restart_num]
            where = where / float(self.restart_interval_length)
        return self._end_value + 0.5 * (self._max_lr - self._end_value) * (
            1 + math.cos(math.pi * where)
        )
