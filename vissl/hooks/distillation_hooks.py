# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
import torch.nn as nn
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from fairscale.nn import FullyShardedDataParallel as FSDP
from vissl.config.attr_dict import AttrDict
from vissl.losses.dino_distillation_loss import DINODistillationLoss
from vissl.losses.distillation_loss import DistillationLoss
from vissl.losses.ibot_distillation_loss import IBOTDistillationLoss
from vissl.losses.msn_distillation_loss import MSNDistillationLoss
from vissl.losses.swav_distillation_loss import SwAVDistillationLoss
from vissl.models import build_model
from vissl.utils.checkpoint import CheckpointLoader
from vissl.utils.fsdp_utils import fsdp_wrapper


class DistillationHook(ClassyHook):
    """
    Hook used to have a teacher model run in parallel of a student model
    in order to perform distillation of the teacher to the student

    The hook contains the teacher model: it runs in evaluation mode and
    provides the soft labels needed to compute the loss of the student

    The hook supports FSDP options to shard the teacher model and avoid
    GPU out of memory due to large models
    """

    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, distillation_config: AttrDict):
        super().__init__()
        self.distillation_config = distillation_config
        self.teacher_config = distillation_config.TEACHER_MODEL
        self.teacher: Optional[nn.Module] = None

    def on_start(self, task: tasks.ClassyTask) -> None:
        if self.distillation_config.eager_teacher_init:
            self._create_teacher_model(task)

    def on_phase_start(self, task: tasks.ClassyTask) -> None:
        """
        Loading the checkpoint and creating the teacher model

        Loading earlier than during the on_forward allows to avoid adding
        a memory spike (due to the broadcasting of the checkpoint) to the
        memory consumption of the activations of the student (which peaks
        when calling on_forward) avoiding CUDA OOM issues
        """
        if self.teacher is None:
            self._create_teacher_model(task)

    def _create_teacher_model(self, task: tasks.ClassyTask) -> None:
        logging.info("Building teacher model")
        self.teacher = build_model(self.teacher_config, task.config["OPTIMIZER"])
        self._shard_if_fsdp_model()
        self.teacher.to(task.device)
        self._initialize_teacher()
        self.teacher.eval()

    def _initialize_teacher(self):
        fake_config = AttrDict({"MODEL": self.teacher_config})
        checkpoint = self._load_teacher_checkpoint()
        self.teacher.init_model_from_weights_params_file(
            config=fake_config, checkpoint=checkpoint, strict=True
        )

    def _load_teacher_checkpoint(self):
        init_weights_path = self.teacher_config["WEIGHTS_INIT"]["PARAMS_FILE"]
        assert init_weights_path, "Teacher model requires some initial weights"
        logging.info(f"Initializing teacher model from: {init_weights_path}")
        return CheckpointLoader.load_and_broadcast_init_weights(
            checkpoint_path=init_weights_path, device=torch.device("cpu")
        )

    def _shard_if_fsdp_model(self):
        """
        For FSDP model, we need a FSDP root to be able to load the checkpoint
        and wrap / synchronize all the FSDP sub-modules
        """
        if any(isinstance(m, FSDP) for m in self.teacher.modules()):
            for module in self.teacher.modules():
                if isinstance(module, FSDP):
                    module._is_root = None
            fsdp_config = self.teacher_config["FSDP_CONFIG"]
            self.teacher = fsdp_wrapper(self.teacher, **fsdp_config)

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Compute the outputs of the teacher model and feed them
        to the distillation loss
        """
        if isinstance(task.loss, DistillationLoss):
            self._send_logits_to_distillation_loss(task)
        elif isinstance(task.loss, SwAVDistillationLoss):
            self._send_prototypes_to_swav_loss(task)
        elif isinstance(task.loss, DINODistillationLoss):
            self._send_prototypes_to_dino_loss(task)
        elif isinstance(task.loss, MSNDistillationLoss):
            self._send_prototypes_to_msn_loss(task)
        elif isinstance(task.loss, IBOTDistillationLoss):
            self._send_prototypes_to_ibot_loss(task)
        else:
            loss_type = type(task.loss)
            message = f"Unknown distillation loss type: {loss_type}"
            raise ValueError(message)

    @torch.no_grad()
    def _send_logits_to_distillation_loss(self, task: tasks.ClassyTask):
        """
        Transmit the teacher logits to the distillation loss
        """
        samples = task.last_batch.sample["input"]
        teacher_logits = self.teacher(samples)[0]

        # TODO - make it better: when distilling SwAV model, we
        #  take the prototype scores as targets
        if isinstance(teacher_logits, list):
            teacher_logits = teacher_logits[-1]

        task.loss.teacher_logits = teacher_logits

    @torch.no_grad()
    def _send_prototypes_to_swav_loss(self, task: tasks.ClassyTask):
        """
        Transmit the teacher logits and SwAV prototypes
        to the distillation loss
        """
        # Only take the bigs crop among the samples to send it through
        # the teacher: these first crops will be used as the assigment
        # for the small crops in the student
        samples = task.last_batch.sample["input"]
        if task.loss.use_two_crops_for_teacher:
            samples = [samples[0], samples[1]]
        else:
            samples = [samples[0]]

        outputs = self.teacher(samples)[0]
        if len(outputs) == 2:
            # SwAV
            teacher_logits, teacher_prototypes_scores = outputs
        else:
            # DINO - with return_embeddings=False  # TODO - improve
            teacher_prototypes_scores = outputs[0]

        # Send the teacher scores (to compute the SwAV assignments) as well
        # as the teacher prototypes to the loss: there is not need to normalize
        # the prototypes in contrast to normal SwAV since the prototypes are
        # never updated on the teacher side (frozen teacher)
        prototypes = self.teacher.heads[0].prototypes0.weight.data
        task.loss.teacher_prototypes = prototypes
        task.loss.teacher_prototypes_scores = teacher_prototypes_scores

    @torch.no_grad()
    def _send_prototypes_to_dino_loss(self, task: tasks.ClassyTask):
        """
        Select the samples to forward to teacher, forward them, and
        send the output to the distillation loss
        """
        samples = task.last_batch.sample["input"]
        samples = [samples[i] for i in range(task.loss.teacher_num_crops)]
        outputs = self.teacher(samples)[0][0]
        task.loss.teacher_scores = outputs

    @torch.no_grad()
    def _send_prototypes_to_msn_loss(self, task: tasks.ClassyTask):
        samples = task.last_batch.sample["input"]
        samples = [samples[i] for i in range(task.loss.teacher_num_crops)]
        outputs = self.teacher(samples)[0]
        task.loss.teacher_probs = outputs

    @torch.no_grad()
    def _send_prototypes_to_ibot_loss(self, task: tasks.ClassyTask):
        """
        Select the samples to forward to teacher, forward them, and
        send the output to the distillation loss
        """
        samples = task.last_batch.sample["input"]
        samples = {"global_views": samples["global_views"]}
        outputs = self.teacher(samples)[0]
        task.loss.teacher_scores = outputs

    @torch.no_grad()
    def on_update(self, task: "tasks.ClassyTask") -> None:
        # Make sure the student uses the same prototypes as the teacher
        if isinstance(task.loss, IBOTDistillationLoss):
            if task.loss.use_teacher_prototypes:
                task.base_model.heads[0].prototypes0.copy_(
                    self.teacher.heads[0].prototypes0
                )
                task.base_model.heads[0].prototypes1.copy_(
                    self.teacher.heads[0].prototypes1
                )
