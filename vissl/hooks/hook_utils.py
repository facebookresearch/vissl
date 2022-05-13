# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math

import torch
import torch.nn as nn
from classy_vision import tasks
from torch.nn.parallel import DistributedDataParallel
from vissl.config import AttrDict
from vissl.models import build_model
from vissl.models.model_helpers import get_no_ddp_model
from vissl.utils.fsdp_utils import fsdp_recursive_reset_lazy_init, fsdp_wrapper


class MomentumTeacherInLossHook:
    """Helper to add to hooks to create and update a momentum encoder in the loss"""

    @staticmethod
    def build_momentum_network(
        task: tasks.ClassyTask,
        config: AttrDict,
    ) -> None:
        """
        Create the teacher: it is an exponential moving average of the student.
        """
        logging.info("Building momentum encoder...")

        # Same architecture but do not apply stochastic depth
        is_fsdp_model = "fsdp" in config.MODEL.TRUNK.NAME
        task.loss.momentum_teacher = build_model(config["MODEL"], config["OPTIMIZER"])
        task.loss.momentum_teacher.to(task.device)

        if not is_fsdp_model:
            # Restore an hypothetical checkpoint
            if task.loss.checkpoint is not None:
                task.loss.load_state_dict(task.loss.checkpoint)
            # Else initialize from the student model
            else:
                task_model = get_no_ddp_model(task.model)
                teacher_model = get_no_ddp_model(task.loss.momentum_teacher)
                teacher_model.load_state_dict(task_model.state_dict())

        # Setup SyncBN (useful for the XCiT)
        task.loss.momentum_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(
            task.loss.momentum_teacher
        )

        # Wrap with DDP (needed for SyncBN) or FSDP
        if is_fsdp_model:
            fsdp_config = task.config["MODEL"]["FSDP_CONFIG"]
            task.loss.momentum_teacher = fsdp_wrapper(
                task.loss.momentum_teacher, **fsdp_config
            )
        else:
            task.loss.momentum_teacher = DistributedDataParallel(
                task.loss.momentum_teacher, device_ids=[task.device]
            )

        if is_fsdp_model:
            # Restore an hypothetical checkpoint
            if task.loss.checkpoint is not None:
                task.loss.load_state_dict(task.loss.checkpoint)
            else:
                # Else initialize from the student model
                task_model = task.base_model
                teacher_model = task.loss.momentum_teacher
                teacher_model.load_local_state_dict(task_model.local_state_dict())
            fsdp_recursive_reset_lazy_init(task.loss.momentum_teacher)

        # no gradients for teacher model
        for p in task.loss.momentum_teacher.parameters():
            p.requires_grad = False

    @staticmethod
    def update_momentum_network(
        task: tasks.ClassyTask,
        init_teacher_momentum: float,
        with_cosine_schedule: bool = True,
    ) -> None:
        """
        EMA update: Each teacher parameter becomes a weighted average
        of its old self and the newest student.
        """
        with torch.no_grad():
            # Compute teacher momentum based on momentum schedule
            if with_cosine_schedule:
                m = 1 - 0.5 * (1 - init_teacher_momentum) * (
                    math.cos(math.pi * task.iteration / task.max_iteration) + 1
                )
            else:
                m = init_teacher_momentum

            # Log the value of the momentum
            task.additional_log_data["teacher_momentum"] = m

            # Access the raw student and teacher models
            task_model = get_no_ddp_model(task.model)
            teacher_model = get_no_ddp_model(task.loss.momentum_teacher)

            # EMA update for the teacher parameters
            for param_q, param_k in zip(
                task_model.parameters(), teacher_model.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
