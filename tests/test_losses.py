# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from collections import namedtuple

import torch
<<<<<<< HEAD
import torch.distributed as dist
import torch.multiprocessing as mp
=======
import torch.nn as nn
>>>>>>> f054707d517ce62caf80d0e13163f36ebf4ca53f
from classy_vision.generic.distributed_util import set_cpu_device
from parameterized import param, parameterized
from vissl.config import AttrDict
from vissl.losses.barlow_twins_loss import BarlowTwinsCriterion
from vissl.losses.cross_entropy_multiple_output_single_target import (
    CrossEntropyMultipleOutputSingleTargetCriterion,
    CrossEntropyMultipleOutputSingleTargetLoss,
)
from vissl.losses.multicrop_simclr_info_nce_loss import MultiCropSimclrInfoNCECriterion
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.losses.swav_loss import SwAVCriterion
<<<<<<< HEAD
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.misc import find_free_tcp_port
=======
>>>>>>> f054707d517ce62caf80d0e13163f36ebf4ca53f


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

    def test_simclr_info_nce_loss(self):
        loss_layer = SimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1
        )
        _ = loss_layer(self._get_embedding())

    def test_multicrop_simclr_info_nce_loss(self):
        loss_layer = MultiCropSimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1, num_crops=NUM_CROPS
        )
        embedding = torch.ones([BATCH_SIZE * NUM_CROPS, EMBEDDING_DIM])
        _ = loss_layer(embedding)

    def test_swav_loss(self):
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

    def test_barlow_twins_loss(self):
        loss_layer = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        _ = loss_layer(self._get_embedding())


class TestBarlowTwinsCriterion(unittest.TestCase):
    """
    Specific tests on Barlow Twins going further than just doing a forward pass
    """

    def test_barlow_twins_backward(self):
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

    @staticmethod
    def worker_fn(gpu_id: int, world_size: int, batch_size: int, port: int):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://0.0.0.0:{port}",
            world_size=world_size,
            rank=gpu_id,
        )
        criterion = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        embeddings = torch.randn(
            (batch_size, EMBEDDING_DIM), dtype=torch.float32, requires_grad=True
        ).cuda()
        criterion(embeddings).backward()

    def test_backward_world_size_1(self):
        if torch.cuda.device_count() >= 1:
            port = find_free_tcp_port()

            WORLD_SIZE = 1
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE, port), nprocs=WORLD_SIZE
            )

    def test_backward_world_size_2(self):
        if torch.cuda.device_count() >= 2:
            port = find_free_tcp_port()

            WORLD_SIZE = 2
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE, port), nprocs=WORLD_SIZE
            )


class TestSimClrCriterion(unittest.TestCase):
    """
    Specific tests on SimCLR going further than just doing a forward pass
    """

    def test_simclr_info_nce_masks(self):
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

    def test_simclr_backward(self):
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

    @staticmethod
    def worker_fn(gpu_id: int, world_size: int, batch_size: int, port: int):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://0.0.0.0:{port}",
            world_size=world_size,
            rank=gpu_id,
        )
        embeddings = torch.full(
            size=(batch_size, 3), fill_value=float(gpu_id), requires_grad=True
        ).cuda(gpu_id)
        gathered = SimclrInfoNCECriterion.gather_embeddings(embeddings)
        if world_size == 1:
            assert gathered.equal(
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=f"cuda:{gpu_id}"
                )
            )
        if world_size == 2:
            assert gathered.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    device=f"cuda:{gpu_id}",
                )
            )
        assert gathered.requires_grad

    def test_gather_embeddings_word_size_1(self):
        if torch.cuda.device_count() >= 1:
            port = find_free_tcp_port()

            WORLD_SIZE = 1
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE, port), nprocs=WORLD_SIZE
            )

    def test_gather_embeddings_word_size_2(self):
        if torch.cuda.device_count() >= 2:
            port = find_free_tcp_port()

            WORLD_SIZE = 2
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE, port), nprocs=WORLD_SIZE
            )


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

    def test_multiple_targets_for_label_smoothing(self):
        targets = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        criterion = CrossEntropyMultipleOutputSingleTargetCriterion()
        expected = (
            (-torch.log(nn.Softmax(dim=-1)(logits)) * targets).sum(dim=1).mean().item()
        )
        self.assertAlmostEqual(criterion(logits, targets).item(), expected)

    def test_label_smoothing_target_transformation(self):
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
