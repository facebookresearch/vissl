# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# We need to do the imports below so that classy vision registry can run
import vissl.data.collators  # NOQA
import vissl.data.ssl_transforms  # NOQA
import vissl.hooks  # NOQA
import vissl.losses  # NOQA
import vissl.meters  # NOQA
import vissl.models  # NOQA
import vissl.models.heads  # NOQA
import vissl.models.trunks  # NOQA
import vissl.optimizers.param_scheduler  # NOQA
from vissl.trainer.train_task import SelfSupervisionTask  # NOQA
from vissl.trainer.trainer_main import SelfSupervisionTrainer  # noqa


__all__ = [k for k in globals().keys() if not k.startswith("_")]
