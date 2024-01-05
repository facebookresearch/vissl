# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from classy_vision.generic.distributed_util import set_cpu_device
from parameterized import param, parameterized
from vissl.config.attr_dict import AttrDict
from vissl.losses.barlow_twins_loss import BarlowTwinsCriterion
from vissl.losses.cross_entropy_multiple_output_single_target import (
    CrossEntropyMultipleOutputSingleTargetCriterion,
    CrossEntropyMultipleOutputSingleTargetLoss,
)
from vissl.losses.distillation_loss import DistillationCriterion, DistillationLoss
from vissl.losses.multicrop_simclr_info_nce_loss import MultiCropSimclrInfoNCECriterion
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.losses.swav_loss import SwAVCriterion


logger = logging.getLogger("__name__")

set_cpu_device()

BATCH_SIZE = 2048
EMBEDDING_DIM = 128
NUM_CROPS = 2
BUFFER_PARAMS_STRUCT = namedtuple(
    "BUFFER_PARAMS_STRUCT", ["effective_batch_size", "world_size", "embedding_dim"]
)
BUFFER_PARAMS = BUFFER_PARAMS_STRUCT(BATCH_SIZE, 1, EMBEDDING_DIM)


class TestLossesForward(unittest.TestCase):
    """
    Minimal testing of the losses: ensure that a forward pass with believable
    dimensions succeeds. This does not make them correct per say.
    """

    @staticmethod
    def _get_embedding():
        return torch.ones([BATCH_SIZE, EMBEDDING_DIM])

    def test_simclr_info_nce_loss(self) -> None:
        loss_layer = SimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1
        )
        _ = loss_layer(self._get_embedding())

    def test_multicrop_simclr_info_nce_loss(self) -> None:
        loss_layer = MultiCropSimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1, num_crops=NUM_CROPS
        )
        embedding = torch.ones([BATCH_SIZE * NUM_CROPS, EMBEDDING_DIM])
        _ = loss_layer(embedding)

    def test_swav_loss(self) -> None:
        loss_layer = SwAVCriterion(
            temperature=0.1,
            crops_for_assign=[0, 1],
            num_crops=2,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[3000],
            local_queue_length=0,
            embedding_dim=EMBEDDING_DIM,
            temp_hard_assignment_iters=0,
            output_dir="",
        )
        _ = loss_layer(scores=self._get_embedding(), head_id=0)

    def test_barlow_twins_loss(self) -> None:
        loss_layer = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        _ = loss_layer(self._get_embedding())


class TestBarlowTwinsCriterion(unittest.TestCase):
    """
    Specific tests on Barlow Twins going further than just doing a forward pass
    """

    def test_barlow_twins_backward(self) -> None:
        EMBEDDING_DIM = 3
        criterion = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        embeddings = torch.randn((4, EMBEDDING_DIM), requires_grad=True)

        self.assertTrue(embeddings.grad is None)
        criterion(embeddings).backward()
        self.assertTrue(embeddings.grad is not None)
        with torch.no_grad():
            next_embeddings = embeddings - embeddings.grad  # gradient descent
            self.assertTrue(criterion(next_embeddings) < criterion(embeddings))


class TestSimClrCriterion(unittest.TestCase):
    """
    Specific tests on SimCLR going further than just doing a forward pass
    """

    def test_simclr_info_nce_masks(self) -> None:
        BATCH_SIZE = 4
        WORLD_SIZE = 2
        buffer_params = BUFFER_PARAMS_STRUCT(
            BATCH_SIZE * WORLD_SIZE, WORLD_SIZE, EMBEDDING_DIM
        )
        criterion = SimclrInfoNCECriterion(buffer_params=buffer_params, temperature=0.1)
        self.assertTrue(
            criterion.pos_mask.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        )
        self.assertTrue(
            criterion.neg_mask.equal(
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            )
        )

    def test_simclr_backward(self) -> None:
        EMBEDDING_DIM = 3
        BATCH_SIZE = 4
        WORLD_SIZE = 1
        buffer_params = BUFFER_PARAMS_STRUCT(
            BATCH_SIZE * WORLD_SIZE, WORLD_SIZE, EMBEDDING_DIM
        )
        criterion = SimclrInfoNCECriterion(buffer_params=buffer_params, temperature=0.1)
        embeddings = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        )

        self.assertTrue(embeddings.grad is None)
        criterion(embeddings).backward()
        self.assertTrue(embeddings.grad is not None)
        print(embeddings.grad)
        with torch.no_grad():
            next_embeddings = embeddings - embeddings.grad  # gradient descent
            self.assertTrue(criterion(next_embeddings) < criterion(embeddings))


class TestCrossEntropyMultipleOutputSingleTargetLoss(unittest.TestCase):
    @parameterized.expand(
        [param(batch_size=1, target_count=2), param(batch_size=16, target_count=10)]
    )
    def test_single_input_single_target(self, batch_size: int, target_count: int):
        torch.random.manual_seed(0)
        logits = torch.randn(size=(batch_size, target_count))
        target = torch.randint(0, target_count, size=(batch_size,))

        ref_criterion = nn.CrossEntropyLoss()
        criterion = CrossEntropyMultipleOutputSingleTargetCriterion()
        self.assertEqual(criterion(logits, target), ref_criterion(logits, target))

    @parameterized.expand(
        [
            param(batch_size=1, target_count=2, input_count=1),
            param(batch_size=16, target_count=10, input_count=2),
        ]
    )
    def test_multiple_inputs_single_target(
        self, batch_size: int, target_count: int, input_count: int
    ):
        torch.random.manual_seed(0)
        logits = [
            torch.randn(size=(batch_size, target_count)) for _ in range(input_count)
        ]
        target = torch.randint(0, target_count, size=(batch_size,))

        ref_criterion = nn.CrossEntropyLoss()
        ref_loss = sum(ref_criterion(logits[i], target) for i in range(input_count))
        criterion = CrossEntropyMultipleOutputSingleTargetCriterion()
        self.assertEqual(criterion(logits, target), ref_loss)

    def test_multiple_targets_for_label_smoothing(self) -> None:
        targets = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        criterion = CrossEntropyMultipleOutputSingleTargetCriterion()
        expected = (
            (-torch.log(nn.Softmax(dim=-1)(logits)) * targets).sum(dim=1).mean().item()
        )
        self.assertAlmostEqual(criterion(logits, targets).item(), expected)

    def test_label_smoothing_target_transformation(self) -> None:
        target = torch.tensor([0, 1, 2], dtype=torch.int64)
        smoothed = (
            CrossEntropyMultipleOutputSingleTargetCriterion.apply_label_smoothing(
                target=target, num_labels=4, label_smoothing=0.1
            )
        )
        expected = torch.tensor(
            [
                [0.9250, 0.0250, 0.0250, 0.0250],
                [0.0250, 0.9250, 0.0250, 0.0250],
                [0.0250, 0.0250, 0.9250, 0.0250],
            ]
        )
        self.assertTrue(torch.allclose(expected, smoothed))

    @parameterized.expand(
        [param(batch_size=1, target_count=2), param(batch_size=16, target_count=10)]
    )
    def test_single_target_label_smoothing(self, batch_size: int, target_count: int):
        torch.random.manual_seed(0)
        logits = torch.randn(size=(batch_size, target_count))
        target = torch.randint(0, target_count, size=(batch_size,))

        # Verify that label smoothing is supported in forward pass
        criterion = CrossEntropyMultipleOutputSingleTargetCriterion(label_smoothing=0.1)
        loss = criterion(logits, target)
        self.assertTrue(loss.item() > 0.0)

    @parameterized.expand(
        [
            param(temperature=0.1, normalize_output=False, label_smoothing=0.0),
            param(temperature=1.0, normalize_output=True, label_smoothing=0.0),
            param(temperature=2.0, normalize_output=False, label_smoothing=0.5),
        ]
    )
    def test_configuration(
        self,
        temperature: float,
        normalize_output: bool,
        label_smoothing: float,
        batch_size: int = 16,
        target_count: int = 10,
    ):
        torch.random.manual_seed(0)
        logits = torch.randn(size=(batch_size, target_count))
        target = torch.randint(0, target_count, size=(batch_size,))
        criterion_ref = CrossEntropyMultipleOutputSingleTargetCriterion(
            temperature=temperature,
            normalize_output=normalize_output,
            label_smoothing=label_smoothing,
        )
        config = AttrDict(
            {
                "temperature": temperature,
                "normalize_output": normalize_output,
                "label_smoothing": label_smoothing,
            }
        )
        criterion = CrossEntropyMultipleOutputSingleTargetLoss(config)
        self.assertEqual(criterion(logits, target), criterion_ref(logits, target))


class TestDistillationCriterion(unittest.TestCase):
    @parameterized.expand([param(temperature=1.0), param(temperature=2.0)])
    def test_hard_criteria_alone(self, temperature: float):
        criterion = DistillationCriterion(
            soft_target_alpha=0.0, temperature=temperature
        )
        logits = torch.tensor([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        teacher_logits = torch.randn(size=(2, 3))
        target = torch.tensor([0, 1], dtype=torch.int64)
        loss = criterion(logits, teacher_logits, target)
        ref_loss = nn.CrossEntropyLoss()(logits, target)
        self.assertAlmostEqual(loss.item(), ref_loss.item(), delta=1e-4)

    @parameterized.expand(
        [
            param(soft_alpha=0.1, temperature=1.0, loss_type="mse"),
            param(soft_alpha=0.5, temperature=5.0, loss_type="kl_divergence"),
            param(soft_alpha=0.9, temperature=9.0, loss_type="cross_entropy"),
        ]
    )
    def test_configuration(self, soft_alpha: float, temperature: float, loss_type: str):
        criterion = DistillationLoss.from_config(
            AttrDict(
                {
                    "soft_target_alpha": soft_alpha,
                    "temperature": temperature,
                    "loss_type": loss_type,
                }
            )
        )
        self.assertEqual(criterion.criterion.hard_target_alpha, 1 - soft_alpha)
        self.assertEqual(criterion.criterion.soft_target_alpha, soft_alpha)
        self.assertEqual(criterion.criterion.temperature, temperature)
        self.assertEqual(criterion.criterion.loss_type.name, loss_type.upper())

    @parameterized.expand(
        [
            param(loss_type="kl_divergence", temperature=10.0),
            param(loss_type="kl_divergence", temperature=1.0),
            param(loss_type="cross_entropy", temperature=10.0),
            param(loss_type="cross_entropy", temperature=1.0),
            param(loss_type="mse", temperature=10.0),
            param(loss_type="mse", temperature=1.0),
        ]
    )
    def test_convergence_with_soft_alpha_only(self, loss_type: str, temperature: float):
        criterion = DistillationLoss.from_config(
            AttrDict(
                {
                    "soft_target_alpha": 1.0,
                    "temperature": temperature,
                    "loss_type": loss_type,
                }
            )
        )

        # An input and its soft target
        x = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]]))
        y = torch.tensor([[3.0, 2.0, 1.0]])
        criterion.teacher_logits = y
        t = torch.tensor([0], dtype=torch.int64)

        # Optimizing the input to imitate soft target
        optimizer = optim.SGD([x], lr=1.0)
        for _ in range(100):
            loss = criterion(x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Assert convergence
        self.assertTrue(torch.allclose(x.data, y.data))
