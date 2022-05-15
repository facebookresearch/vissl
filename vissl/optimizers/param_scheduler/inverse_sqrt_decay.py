# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    register_param_scheduler,
    UpdateInterval,
)


@register_param_scheduler("inverse_sqrt")
class InverseSqrtScheduler(ClassyParamScheduler):
    """
    Decay the LR based on the inverse square root of the update number.

    Example:

        .. code-block:: python

            start_value: 4.8
            warmup_interval_length: 0.1
    Corresponds to a inverse sqrt decay schedule with values in [4.8, 0]
    """

    def __init__(
        self,
        start_value: float,
        warmup_interval_length: float,
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        super().__init__(update_interval=update_interval)
        self._start_value = start_value
        self.warmup_interval_length = warmup_interval_length

        self.decay_factor = self._start_value
        if self.warmup_interval_length > 0.0:
            self.decay_factor = self._start_value * self.warmup_interval_length**0.5

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "InverseSqrtScheduler":
        """
        Instantiates a InverseSqrtScheduler from a configuration.

        Args:
            config: A configuration for a InverseSqrtScheduler.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A InverseSqrtScheduler instance.
        """
        assert "start_value" in config, "InverseSqrtScheduler requires a start_value"
        assert (
            "warmup_interval_length" in config
        ), "InverseSqrtScheduler requires a warmup_interval_length"

        return cls(
            start_value=config["start_value"],
            warmup_interval_length=config["warmup_interval_length"],
            update_interval=UpdateInterval.from_config(config, UpdateInterval.STEP),
        )

    def __call__(self, where: float):
        if where > 0.0:
            return self.decay_factor * (where**-0.5)
        else:
            return self.decay_factor
