# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest

import torch
import torch.multiprocessing as mp
from classy_vision.optim import build_optimizer
from torch.nn.parallel import DistributedDataParallel
from vissl.losses.swav_loss import SwAVLoss
from vissl.models import build_model
from vissl.optimizers import *  # noqa
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import gpu_test, init_distributed_on_file, with_temp_files


class TestRegnetFSDP(unittest.TestCase):
    """
    Test the Regnet FSDP model in comparison with the DDP Regnet
    to verify that both converge to the same losses
    """

    @staticmethod
    def _create_pretraining_config(
        with_fsdp: bool, with_activation_checkpointing: bool, with_larc: bool
    ):
        cfg = compose_hydra_configuration(
            [
                "config=pretrain/swav/swav_8node_resnet",
                "+config/pretrain/swav/models=regnet16Gf",
                "config.SEED_VALUE=2",
                "config.MODEL.AMP_PARAMS.USE_AMP=True",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                f"config.OPTIMIZER.use_larc={with_larc}",
                "config.LOSS.swav_loss.epsilon=0.03",
                "config.MODEL.FSDP_CONFIG.flatten_parameters=True",
                "config.MODEL.FSDP_CONFIG.mixed_precision=False",
                "config.MODEL.FSDP_CONFIG.fp32_reduce_scatter=False",
            ],
        )
        args, config = convert_to_attrdict(cfg)
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head_fsdp"
        else:
            config["MODEL"]["TRUNK"]["NAME"] = "regnet_v2"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "swav_head"

        if with_larc and with_fsdp:
            config.MODEL.FSDP_CONFIG.flatten_parameters = False
            config.OPTIMIZER.name = "sgd_fsdp"

        config["MODEL"]["ACTIVATION_CHECKPOINTING"][
            "USE_ACTIVATION_CHECKPOINTING"
        ] = with_activation_checkpointing
        return config

    @staticmethod
    def _pretraining_worker(
        gpu_id: int,
        with_fsdp: bool,
        with_activation_checkpointing: bool,
        with_larc: bool,
        sync_file: str,
        result_file: str,
    ):
        init_distributed_on_file(world_size=2, gpu_id=gpu_id, sync_file=sync_file)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True

        # Create the inputs
        batch = torch.randn(size=(8, 3, 224, 224)).cuda()
        target = torch.tensor(0.0).cuda()

        # Create a fake model based on SWAV blocks
        config = TestRegnetFSDP._create_pretraining_config(
            with_fsdp, with_activation_checkpointing, with_larc=with_larc
        )
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        model = model.cuda()
        if with_fsdp:
            model = fsdp_wrapper(model, **config.MODEL.FSDP_CONFIG)
        else:
            model = DistributedDataParallel(model, device_ids=[gpu_id])
        criterion = SwAVLoss(loss_config=config["LOSS"]["swav_loss"])
        optimizer = build_optimizer(config["OPTIMIZER"])
        optimizer.set_param_groups(model.parameters())

        # Run a few iterations and collect the losses
        losses = []
        num_iterations = 5
        for iteration in range(num_iterations):
            out = model(batch)
            loss = criterion(out[0], target)
            if gpu_id == 0:
                losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if iteration <= 2:
                for name, param in model.named_parameters():
                    if "prototypes" in name:
                        param.grad = None
            optimizer.step(where=float(iteration / num_iterations))

        # Store the losses in a file to compare several methods
        if gpu_id == 0:
            with open(result_file, "wb") as f:
                pickle.dump(losses, f)

    @staticmethod
    def run_pretraining(
        with_fsdp: bool,
        with_checkpointing: bool,
        with_larc: bool,
        output_file_name: str,
    ):
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                TestRegnetFSDP._pretraining_worker,
                (with_fsdp, with_checkpointing, with_larc, sync_file, output_file_name),
                nprocs=2,
            )

    @gpu_test(gpu_count=2)
    def test_regnet_fsdp_convergence_on_swav(self) -> None:
        """
        Run SWAV architecture with DDP or with FSDP with or without
        activation checkpointing and check that the results match
        """
        with with_temp_files(count=3) as file_names:
            self.run_pretraining(
                with_fsdp=False,
                with_checkpointing=False,
                with_larc=False,
                output_file_name=file_names[0],
            )
            self.run_pretraining(
                with_fsdp=True,
                with_checkpointing=False,
                with_larc=False,
                output_file_name=file_names[1],
            )
            self.run_pretraining(
                with_fsdp=True,
                with_checkpointing=True,
                with_larc=False,
                output_file_name=file_names[2],
            )

            results = []
            for file_name in file_names:
                with open(file_name, "rb") as f:
                    result = pickle.load(f)
                    results.append(result)
            self.assertEqual(results[0], results[1], "DDP vs FSDP")
            self.assertEqual(results[1], results[2], "Activation checkpointing")

    @gpu_test(gpu_count=2)
    def test_regnet_fsdp_convergence_on_swav_with_larc(self) -> None:
        """
        Run SWAV architecture with DDP or with FSDP with or without
        activation checkpointing and check that the results match
        """
        with with_temp_files(count=2) as file_names:
            self.run_pretraining(
                with_fsdp=False,
                with_checkpointing=False,
                with_larc=True,
                output_file_name=file_names[0],
            )
            self.run_pretraining(
                with_fsdp=True,
                with_checkpointing=False,
                with_larc=True,
                output_file_name=file_names[1],
            )

            results = []
            for file_name in file_names:
                with open(file_name, "rb") as f:
                    result = pickle.load(f)
                    # TODO (Quentin) - figure out why it diverges slightly after a while
                    result[3] = round(result[3], 5)
                    result[4] = round(result[4], 4)
                    results.append(result)

            self.assertEqual(
                len(results[0]), len(results[1]), "DDP vs FSDP (LARC) Loss Lengths"
            )

            for i, ddp_result in enumerate(results[0]):
                fsdp_result = results[1][i]
                self.assertAlmostEqual(ddp_result, fsdp_result, places=4)
