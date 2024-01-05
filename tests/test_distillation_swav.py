# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
import torch.nn.functional as functional
from vissl.losses.swav_distillation_loss import SwAVDistillationCriterion
from vissl.losses.swav_loss import SwAVCriterion
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
    spawn_distributed_test,
)


class TestDistillationSwAV(unittest.TestCase):
    @staticmethod
    def _create_swav_pretraining_config(num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.SEED_VALUE=0",
                "config.LOSS.swav_loss.epsilon=0.03",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_soft_swav_distillation_config(checkpoint_path: str, num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_distillation_swav_2gpus",
                f"config.DISTILLATION.TEACHER_MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.SEED_VALUE=0",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
                "config.OPTIMIZER.num_epochs=2",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=False",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _test_swav_distillation_criterion_worker(gpu_id: int):
        torch.random.manual_seed(gpu_id)

        num_crops = 6
        batch_size = 5
        embedding_dim = 16
        num_prototypes = 32
        criterion = SwAVDistillationCriterion(
            temperature=0.1,
            num_crops=num_crops,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            output_dir=".",
            temp_hard_assignment_iters=0,
            local_queue_length=30,
            swapped_assignment=True,
        ).cuda(gpu_id)
        logits = torch.randn(size=(batch_size * num_crops, embedding_dim)).cuda(gpu_id)
        teacher_scores = torch.randn(size=(batch_size, num_prototypes)).cuda(gpu_id)
        teacher_prototypes = torch.randn(size=(num_prototypes, embedding_dim)).cuda(
            gpu_id
        )
        loss = criterion(logits, teacher_scores, teacher_prototypes)
        print(f"DISTILL LOSS (GPU ID {gpu_id}):", loss.item())

        # Create a SwAV criterion that mimics the one of the SwAV loss criterion:
        # - only once crop to assign (otherwise same parameters)
        # - then places the teacher scores as first crop score
        ref_criterion = SwAVCriterion(
            temperature=0.1,
            crops_for_assign=[0],
            num_crops=num_crops,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            local_queue_length=0,
            embedding_dim=embedding_dim,
            temp_hard_assignment_iters=0,
            output_dir=".",
        ).cuda(gpu_id)
        scores = functional.linear(logits, teacher_prototypes)
        scores[:batch_size] = teacher_scores
        ref_loss = ref_criterion(scores, head_id=0)
        print(f"REF LOSS (GPU ID {gpu_id}):", ref_loss.item())

        message = f"REF LOSS {ref_loss.item()} VS LOSS {loss.item()}"
        assert round(loss.item(), 5) == round(ref_loss.item(), 5), message

    @gpu_test(gpu_count=2)
    def test_swav_distillation_criterion(self) -> None:
        spawn_distributed_test(
            gpu_count=2, worker_fn=self._test_swav_distillation_criterion_worker
        )

    @staticmethod
    def _test_swav_distillation_criterion_two_large_crops_worker(gpu_id: int):
        torch.random.manual_seed(gpu_id)

        num_crops = 6
        batch_size = 5
        embedding_dim = 16
        num_prototypes = 32
        criterion = SwAVDistillationCriterion(
            temperature=0.1,
            num_crops=num_crops,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            output_dir=".",
            temp_hard_assignment_iters=0,
            local_queue_length=30,
            swapped_assignment=True,
        ).cuda(gpu_id)
        logits = torch.randn(size=(batch_size * num_crops, embedding_dim)).cuda(gpu_id)
        teacher_scores = torch.randn(size=(batch_size * 2, num_prototypes)).cuda(gpu_id)
        teacher_prototypes = torch.randn(size=(num_prototypes, embedding_dim)).cuda(
            gpu_id
        )
        loss = criterion(logits, teacher_scores, teacher_prototypes)
        print(f"DISTILL LOSS (GPU ID {gpu_id}):", loss.item())

        # Create a SwAV criterion that mimics the one of the SwAV loss criterion:
        # - only once crop to assign (otherwise same parameters)
        # - then places the teacher scores as first crop score
        ref_loss = torch.tensor(0.0).cuda(gpu_id)
        for i in range(2):
            ref_criterion = SwAVCriterion(
                temperature=0.1,
                crops_for_assign=[i],
                num_crops=num_crops,
                num_iters=3,
                epsilon=0.05,
                use_double_prec=False,
                num_prototypes=[num_prototypes],
                local_queue_length=0,
                embedding_dim=embedding_dim,
                temp_hard_assignment_iters=0,
                output_dir=".",
            ).cuda(gpu_id)
            scores = functional.linear(logits, teacher_prototypes)
            scores[i * batch_size : (i + 1) * batch_size] = teacher_scores[
                i * batch_size : (i + 1) * batch_size
            ]
            ref_loss += ref_criterion(scores, head_id=0)
        ref_loss = ref_loss / 2
        print(f"REF LOSS (GPU ID {gpu_id}):", ref_loss.item())

        message = f"REF LOSS {ref_loss.item()} VS LOSS {loss.item()}"
        assert round(loss.item(), 5) == round(ref_loss.item(), 5), message

    @gpu_test(gpu_count=2)
    def test_swav_distillation_criterion_two_large_crops(self) -> None:
        spawn_distributed_test(
            gpu_count=2,
            worker_fn=self._test_swav_distillation_criterion_two_large_crops_worker,
        )

    @staticmethod
    def _test_swav_distillation_criterion_queue_worker(gpu_id: int):
        """
        Verify that we can use the queue of the SwAV distillation
        criterion to increase the stability of the assignments
        """
        torch.random.manual_seed(gpu_id)

        num_crops = 6
        batch_size = 5
        embedding_dim = 16
        num_prototypes = 32
        criterion = SwAVDistillationCriterion(
            temperature=0.1,
            num_crops=num_crops,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            output_dir=".",
            temp_hard_assignment_iters=0,
            local_queue_length=30,
            swapped_assignment=True,
        ).cuda(gpu_id)

        logits = torch.randn(size=(batch_size * num_crops, embedding_dim)).cuda(gpu_id)
        teacher_prototypes = torch.randn(size=(num_prototypes, embedding_dim)).cuda(
            gpu_id
        )
        teacher_scores_1 = torch.rand(size=(batch_size, num_prototypes)).cuda(gpu_id)
        teacher_scores_2 = torch.rand(size=(batch_size, num_prototypes)).cuda(gpu_id)
        for i in range(batch_size):
            teacher_scores_1[i, i + 2 * batch_size * gpu_id] += 10
            teacher_scores_2[i, i + 2 * batch_size * gpu_id + batch_size] += 10

        # Compute loss with no previous scores in the queue
        loss_1 = criterion(logits, teacher_scores_1, teacher_prototypes)
        asssignments_1 = criterion.compute_teacher_assignements(teacher_scores_1)
        print(loss_1.item(), "-", asssignments_1.max().item())

        # Add some previous teacher scores in the queue
        criterion.add_to_teacher_scores_queue(teacher_scores_2)
        criterion.use_queue = True

        # Compute loss with this queue
        loss_2 = criterion(logits, teacher_scores_1, teacher_prototypes)
        asssignments_2 = criterion.compute_teacher_assignements(teacher_scores_1)
        print(loss_2.item(), "-", asssignments_2.max().item())

        # Check that the assignments are less spread (more certain)
        assert asssignments_2.max().item() > asssignments_1.max().item()
        assert loss_1.item() != loss_2.item(), "Loss should be different"

    @gpu_test(gpu_count=2)
    def test_swav_distillation_criterion_queue(self) -> None:
        spawn_distributed_test(
            gpu_count=2, worker_fn=self._test_swav_distillation_criterion_queue_worker
        )

    @staticmethod
    def _test_swav_distillation_criterion_no_shk(gpu_id: int):
        torch.random.manual_seed(gpu_id)

        num_crops = 6
        batch_size = 5
        embedding_dim = 16
        num_prototypes = 300
        criterion = SwAVDistillationCriterion(
            temperature=0.5,
            num_crops=num_crops,
            num_iters=0,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            output_dir=".",
            temp_hard_assignment_iters=0,
            local_queue_length=30,
            swapped_assignment=True,
        ).cuda(gpu_id)

        logits = torch.randn(size=(batch_size * num_crops, embedding_dim)).cuda(gpu_id)
        teacher_scores = torch.randn(size=(batch_size * 2, num_prototypes)).cuda(gpu_id)
        teacher_prototypes = torch.randn(size=(num_prototypes, embedding_dim)).cuda(
            gpu_id
        )
        loss = criterion(logits, teacher_scores, teacher_prototypes)
        print(f"DISTILL LOSS (GPU ID {gpu_id}):", loss.item())

    @gpu_test(gpu_count=2)
    def test_swav_distillation_criterion_no_shk(self) -> None:
        spawn_distributed_test(
            gpu_count=2, worker_fn=self._test_swav_distillation_criterion_no_shk
        )

    @staticmethod
    def _test_swav_distillation_criterion_no_teacher_prototypes(gpu_id: int):
        torch.random.manual_seed(gpu_id)

        num_crops = 6
        batch_size = 5
        num_prototypes = 300
        criterion = SwAVDistillationCriterion(
            temperature=0.5,
            num_crops=num_crops,
            num_iters=0,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[num_prototypes],
            output_dir=".",
            temp_hard_assignment_iters=0,
            local_queue_length=30,
            swapped_assignment=True,
        ).cuda(gpu_id)

        logits = torch.randn(size=(batch_size * num_crops, num_prototypes)).cuda(gpu_id)
        teacher_scores = torch.randn(size=(batch_size * 2, num_prototypes)).cuda(gpu_id)
        loss = criterion(logits, teacher_scores, None)
        print(f"DISTILL LOSS (GPU ID {gpu_id}):", loss.item())

    @gpu_test(gpu_count=2)
    def test_swav_distillation_criterion_no_teacher_prototypes(self) -> None:
        spawn_distributed_test(
            gpu_count=2,
            worker_fn=self._test_swav_distillation_criterion_no_teacher_prototypes,
        )

    @gpu_test(gpu_count=2)
    def test_swav_soft_distillation(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_swav_pretraining_config(num_gpu=2)
            run_integration_test(pretrain_config)
            checkpoint_path = os.path.join(pretrain_dir, "checkpoint.torch")

            # Distillation, using 1 crop on the teacher side
            with in_temporary_directory():
                distill_config = self._create_soft_swav_distillation_config(
                    checkpoint_path=checkpoint_path, num_gpu=2
                )
                distill_config.LOSS.swav_distillation_loss.normalize_student_feats = (
                    False
                )
                result = run_integration_test(distill_config)
                losses = result.get_losses()
                print(losses)
                self.assertTrue(10, len(losses))
                self.assertGreater(losses[0], losses[-1])

            # Distillation, using 2 crops on the teacher side
            with in_temporary_directory():
                distill_config = self._create_soft_swav_distillation_config(
                    checkpoint_path=checkpoint_path, num_gpu=2
                )
                distill_config.LOSS.swav_distillation_loss.normalize_student_feats = (
                    False
                )
                distill_config.LOSS.swav_distillation_loss.use_two_crops_for_teacher = (
                    True
                )
                distill_config.LOSS.swav_distillation_loss.swapped_assignment = True
                result = run_integration_test(distill_config)
                losses = result.get_losses()
                print(losses)
                self.assertTrue(10, len(losses))
                self.assertGreater(losses[0], losses[-1])

            # Distillation:
            # - using 2 crops on the teacher side
            # - without SHK
            # - without teacher prototypes
            with in_temporary_directory():
                distill_config = self._create_soft_swav_distillation_config(
                    checkpoint_path=checkpoint_path, num_gpu=2
                )
                distill_config.MODEL.HEAD.PARAMS[0][1]["dims"][2] = 3000
                distill_config.LOSS.swav_distillation_loss.normalize_student_feats = (
                    False
                )
                distill_config.LOSS.swav_distillation_loss.use_teacher_prototypes = (
                    False
                )
                distill_config.LOSS.swav_distillation_loss.use_two_crops_for_teacher = (
                    True
                )
                distill_config.LOSS.swav_distillation_loss.swapped_assignment = False

                result = run_integration_test(distill_config)
                losses = result.get_losses()
                print(losses)
                self.assertTrue(10, len(losses))
                self.assertGreater(losses[0], losses[-1])

            # Distillation, using normalisation of student output features
            with in_temporary_directory():
                distill_config = self._create_soft_swav_distillation_config(
                    checkpoint_path=checkpoint_path, num_gpu=2
                )
                distill_config.LOSS.swav_distillation_loss.normalize_student_feats = (
                    True
                )
                result = run_integration_test(distill_config)
                losses = result.get_losses()
                print(losses)
                self.assertTrue(10, len(losses))
                self.assertGreater(losses[0], losses[-1])
