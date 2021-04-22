# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pickle
import tempfile
import unittest
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from hydra.experimental import compose, initialize_config_module
from torch.nn.parallel import DistributedDataParallel
from vissl.losses.swav_loss import SwAVLoss
from vissl.models import build_model
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.hydra_config import convert_to_attrdict


class TestRegnetFSDP(unittest.TestCase):
    """
    Test the Regnet FSDP model in comparison with the DDP Regnet
    to verify that both converge to the same losses
    """

    @staticmethod
    def _create_config(with_fsdp: bool):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=pretrain/swav/swav_8node_resnet",
                    "+config/pretrain/swav/models=regnet16Gf",
                    "config.SEED_VALUE=2",
                    "config.MODEL.AMP_PARAMS.USE_AMP=True",
                    "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                    "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=True",
                    "config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch",
                    "config.OPTIMIZER.num_epochs=1",
                    "config.OPTIMIZER.use_larc=False",
                    "config.LOSS.swav_loss.epsilon=0.03",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=16",
                    "config.DISTRIBUTED.NCCL_DEBUG=False",
                    "config.DISTRIBUTED.NUM_NODES=1",
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
        return config

    @staticmethod
    def _distributed_worker(
        gpu_id: int, with_fsdp: bool, sync_file: str, result_file: str
    ):
        torch.cuda.set_device(gpu_id)
        dist.init_process_group(
            backend="nccl", init_method="file://" + sync_file, world_size=2, rank=gpu_id
        )

        # Create the inputs
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        batch = torch.randn(size=(8, 3, 224, 224)).cuda()

        # Create a fake model based on SWAV blocks
        config = TestRegnetFSDP._create_config(with_fsdp)
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        model = model.cuda()
        if with_fsdp:
            model = fsdp_wrapper(model, **config.MODEL.FSDP_CONFIG)
        else:
            model = DistributedDataParallel(model, device_ids=[gpu_id])
        criterion = SwAVLoss(loss_config=config["LOSS"]["swav_loss"])
        optimizer = optim.SGD(model.parameters(), lr=1e-2)

        # Run a few iterations and collect the losses
        losses = []
        for iteration in range(5):
            out = model(batch)
            loss = criterion(out[0], torch.tensor(0.0).cuda())
            if gpu_id == 0:
                losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if iteration <= 2:
                for name, param in model.named_parameters():
                    if "prototypes" in name:
                        param.grad = None
            optimizer.step()

        # Store the losses in a file to compare several methods
        if gpu_id == 0:
            with open(result_file, "wb") as f:
                pickle.dump(losses, f)

    @staticmethod
    @contextmanager
    def _with_temp_files(count: int):
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])

    def test_regnet_fsdp_convergence_on_swav(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs to run the test")
            return

        # Run with and without FSDP and check that the results match
        with TestRegnetFSDP._with_temp_files(count=4) as file_names:
            with_fsdp = False
            mp.spawn(
                TestRegnetFSDP._distributed_worker,
                (with_fsdp, file_names[0], file_names[1]),
                nprocs=2,
            )
            with_fsdp = True
            mp.spawn(
                TestRegnetFSDP._distributed_worker,
                (with_fsdp, file_names[2], file_names[3]),
                nprocs=2,
            )
            with open(file_names[1], "rb") as f:
                ddp_result = pickle.load(f)
            with open(file_names[3], "rb") as f:
                fsdp_result = pickle.load(f)
            self.assertEqual(ddp_result, fsdp_result)
