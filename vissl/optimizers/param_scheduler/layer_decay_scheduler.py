# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

from classy_vision.optim.param_scheduler import (
    ClassyParamScheduler,
    ConstantParamScheduler,
    UpdateInterval,
)


class LayerDecayScheduler(ClassyParamScheduler):
    """
    Parameter scheduler decorator used in conjunction with layer decay
    to reduce the learning rate of some specific parameter groups
    """

    def __init__(
        self,
        wrapped: Union[float, ClassyParamScheduler],
        lr_decay: float,
        update_interval: UpdateInterval = UpdateInterval.STEP,
    ):
        super().__init__(update_interval)
        self.wrapped: ClassyParamScheduler = self._to_param_scheduler(wrapped)
        self.lr_decay = lr_decay

    @staticmethod
    def _to_param_scheduler(wrapped):
        return (
            wrapped
            if isinstance(wrapped, ClassyParamScheduler)
            else ConstantParamScheduler(wrapped)
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LayerDecayScheduler":
        pass  # Never instantiated from config

    def __call__(self, where: float):
        return self.wrapped(where) * self.lr_decay

    def __str__(self):
        return f"LayerDecayScheduler({self.lr_decay}, {self.wrapped})"
