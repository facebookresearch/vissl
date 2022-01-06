# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List, Union

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook


class ModelOutputMaskHook(ClassyHook):
    """
    This hook is used when config.METERS:.model_output_mask=True. It is used to ignore
    certain classes during test-time and is used for Imagenet out-of-domain datasets
    (e.g. objectnet and imagenet_r).

    The logic is as follows:

    1. Find all unique label indexes in the test set.
    2. During testing, set model output to -inf when the label index is not present, leave it
    unchanged otherwise.
    3. Feed model output into the respective meters. For classification with logit outputs,
    this has the effect of ignoring the classes that are not present in the
    top-1/top-5 calculations.
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self):
        super().__init__()
        self.unique_labels = None
        self.model_output_masks = None

    def on_forward(self, task: tasks.ClassyTask) -> None:
        assert (
            task.config.METERS.model_output_mask
        ), "This hook should only be run when using a model_output_mask"

        if task.train:
            # Hook will do nothing when training.
            return

        assert (
            "test" in task.datasets
        ), "Task must have test dataset in order to use the ModelOutputMaskHook."

        if self.unique_labels is None:
            self.unique_labels = self._get_unique_labels(task)

        # Get the model output(s)
        model_outputs = task.last_batch.model_output
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]

        # Mask the model output in-place.
        model_output_masks = self._create_model_output_masks(model_outputs)
        for out, mask in zip(model_outputs, model_output_masks):
            out.masked_fill_(mask, -math.inf)

    def _create_model_output_masks(
        self, model_outputs: Union[torch.Tensor, List[torch.Tensor]]
    ):
        if self.model_output_masks is not None:
            return self.model_output_masks

        # Model output can be a list of tensors or a tensor.
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]

        # Read the labels present and create the model_output_mask
        masks = []
        for model_output in model_outputs:
            mask = []
            for i in range(model_output.shape[-1]):
                mask.append(i not in self.unique_labels)
            mask = torch.tensor(mask, device=model_output.device)
            masks.append(mask)

        self.model_output_masks = masks
        return self.model_output_masks

    def _get_unique_labels(self, task: tasks.ClassyTask):
        # Get all unique labels for mask.
        unique_labels = set()

        # Load labels if they are not already loaded.
        test_dataset = task.datasets["test"]
        if not test_dataset.labels_init:
            test_dataset.load_labels()

        for labels in test_dataset.label_objs:
            # labels can either be an array or, in case of multi-output,
            # an array of arrays.
            if isinstance(labels[0], list):
                for l in labels:
                    unique_labels = unique_labels.union(set(l))
            else:
                unique_labels = unique_labels.union(set(labels))

        return unique_labels
