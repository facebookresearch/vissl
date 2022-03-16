# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
published under an Apache License 2.0, with modifications by Matthew Leavitt
(ito@fb.com; matthew.l.leavitt@gmail.com). Modifications are described here and
notated where present in the code.

Modifications:
- Skip keys EMA where dtype is not torch.float32.

COMMENT FROM ORIGINAL:
Exponential Moving Average (EMA) of model updates
Hacked together by / Copyright 2020 Ross Wightman
"""
import logging

import torch
import torch.nn as nn
from classy_vision import models
from classy_vision.generic.distributed_util import is_primary


class ModelEmaV2(nn.Module):
    """
    Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually
    in a separate process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self, model: "models.ClassyModel", decay: float = 0.9999, device: str = "gpu"
    ):
        super(ModelEmaV2, self).__init__()
        self.module = model
        # Set ema model to eval mode. This will for example use the running batchnorm statistics
        # for use in BatchNorm layers.
        self.module.eval()
        self.decay = decay
        self.first_run = True
        # perform ema on different device from model if set
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.module = self.module.to(device=self.device)

    @torch.no_grad()
    def _update(self, model, update_fn):
        base_model_state_dict = model.state_dict()
        for key, ema_params in self.module.state_dict().items():
            model_params = base_model_state_dict[key]

            if self.device is not None:
                model_params = model_params.to(device=self.device)
            if ema_params.dtype != torch.float32:
                # This is modification from original code.
                if self.first_run and is_primary():
                    logging.warning(
                        f"EMA: will be skipping key: {key} since it is of type: {ema_params.dtype}"  # NOQA
                    )
                value = model_params
            else:
                value = update_fn(ema_params, model_params)
            ema_params.copy_(value)

        self.first_run = False

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
