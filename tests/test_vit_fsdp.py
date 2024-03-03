# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import pickle
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from vissl.config import AttrDict
from vissl.models import build_model
from vissl.models.heads import DINOHead
from vissl.models.heads.dino_head import DINOHeadFSDP
from vissl.models.trunks.vision_transformer import Block, PatchEmbed
from vissl.utils.fsdp_utils import fsdp_wrapper, is_valid_fsdp_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.misc import torch_version
from vissl.utils.test_utils import gpu_test, init_distributed_on_file, with_temp_files


class TestVitFSDP(unittest.TestCase):
    """
    ---------------------------------------------------------------------------
    Testing ViT individual blocks
    ---------------------------------------------------------------------------
    """

    @gpu_test(gpu_count=2)
    def test_blocks_fsdp_vs_ddp_convergence(self) -> None:
        with_amp = True
        with with_temp_files(count=2) as file_names:
            self._run_block_training_loop(
                with_fsdp=True, with_amp=with_amp, output_file_name=file_names[0]
            )
            self._run_block_training_loop(
                with_fsdp=False, with_amp=with_amp, output_file_name=file_names[1]
            )

            results = []
            for file_name in file_names:
                with open(file_name, "rb") as f:
                    result = pickle.load(f)
                    results.append(result)
            self.assertEqual(results[0], results[1], "DDP vs FSDP")

    @classmethod
    def _run_block_training_loop(
        cls, with_fsdp: bool, with_amp: bool, output_file_name: str
    ):
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                cls._block_worker,
                (with_fsdp, with_amp, sync_file, output_file_name),
                nprocs=2,
            )

    @staticmethod
    def _block_worker(
        gpu_id: int, with_fsdp: bool, with_amp: bool, sync_file: str, result_file: str
    ):
        init_distributed_on_file(world_size=2, gpu_id=gpu_id, sync_file=sync_file)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()

        # Create the inputs
        batch_size = 8
        embed_dim = 384
        batch = torch.randn(size=(batch_size, 3, 224, 224)).cuda()

        # Create the model
        num_blocks = 5
        patch_embed = PatchEmbed(embed_dim=embed_dim).cuda()
        blocks = [Block(dim=embed_dim, num_heads=6).cuda() for _ in range(num_blocks)]
        norm = nn.LayerNorm(embed_dim).cuda()

        # Wrap the model with FSDP or DDP
        if with_fsdp:
            fsdp_config = {
                "flatten_parameters": True,
                "mixed_precision": with_amp,
                "fp32_reduce_scatter": False,  # Only makes sense to be True when mixed_precision is True.
                "compute_dtype": torch.float32,
                "bucket_cap_mb": 0,
                "clear_autocast_cache": True,
                "verbose": True,
                "reshard_after_forward": True,
            }
            blocks = [fsdp_wrapper(block, **fsdp_config) for block in blocks]
            model = nn.Sequential(patch_embed, *blocks, norm)
            model = fsdp_wrapper(model, **fsdp_config)
        else:
            model = nn.Sequential(patch_embed, *blocks, norm)
            model = DistributedDataParallel(model, device_ids=[gpu_id])

        # Print the model
        if gpu_id == 0:
            print(model)

        # Create the optimizer
        param_groups = [
            {
                "params": model.parameters(),
                "lr": 1e-4,
                "weight_decay": 1e-3,
            }
        ]
        optimizer = optim.AdamW(param_groups)

        # Go through several training loops
        losses = []
        for step in range(5):

            # Setup the AMP context if necessary
            context = contextlib.suppress()
            if with_amp:
                context = torch.cuda.amp.autocast()

            # Forward pass
            with context:
                out = model(batch)
                out = out.mean()

            # Backward pass
            if torch_version() >= (1, 7, 0):
                model.zero_grad(set_to_none=True)
            else:
                model.zero_grad()
            out.backward()
            optimizer.step()

            # Report results and run schedulers
            torch.distributed.all_reduce(out)
            losses.append(out.item())
            optimizer.param_groups[0].update(
                {
                    "params": model.parameters(),
                    "lr": 1e-4 + step * 1e-4,
                    "weight_decay": 1e-3 + step * 1e-3,
                }
            )

            # Report memory usage
            if gpu_id == 0:
                print(torch.cuda.max_memory_allocated() // 1e6)

        # Dump the list of losses
        if gpu_id == 0:
            print(losses)
            with open(result_file, "wb") as f:
                pickle.dump(losses, f)

    """
    ---------------------------------------------------------------------------
    Testing DINO Head FSDP
    ---------------------------------------------------------------------------
    """

    @gpu_test(gpu_count=2)
    def test_dino_head_fsdp(self) -> None:
        with_amp = False
        with with_temp_files(count=2) as file_names:
            self._run_dino_head_loop(
                with_fsdp=True, with_amp=with_amp, output_file_name=file_names[0]
            )
            self._run_dino_head_loop(
                with_fsdp=False, with_amp=with_amp, output_file_name=file_names[1]
            )

            results = []
            for file_name in file_names:
                with open(file_name, "rb") as f:
                    result = pickle.load(f)
                    results.append(result)
            self.assertEqual(results[0], results[1], "DDP vs FSDP")

    @classmethod
    def _run_dino_head_loop(
        cls, with_fsdp: bool, with_amp: bool, output_file_name: str
    ):
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                cls._dino_head_worker,
                (with_fsdp, with_amp, sync_file, output_file_name),
                nprocs=2,
            )

    @staticmethod
    def _dino_head_worker(
        gpu_id: int, with_fsdp: bool, with_amp, sync_file: str, result_file: str
    ):
        init_distributed_on_file(world_size=2, gpu_id=gpu_id, sync_file=sync_file)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()

        # Create the inputs
        batch_size = 8
        embed_dim = 4
        bottleneck_dim = 5
        num_clusters = 16
        batch = torch.randn(size=(batch_size, embed_dim)).cuda()

        model_config = AttrDict(
            {
                "FSDP_CONFIG": {
                    "flatten_parameters": True,
                    "mixed_precision": with_amp,
                    "fp32_reduce_scatter": False,  # Only makes sense to be True when mixed_precision is True.
                    "compute_dtype": torch.float32,
                    "bucket_cap_mb": 0,
                    "clear_autocast_cache": True,
                    "verbose": True,
                    "reshard_after_forward": True,
                }
            }
        )

        # Create the model
        normalize_last_layer = True
        if with_fsdp:
            model = DINOHeadFSDP(
                model_config=model_config,
                in_dim=embed_dim,
                num_clusters=[num_clusters],
                bottleneck_dim=bottleneck_dim,
                normalize_last_layer=normalize_last_layer,
            ).cuda()
            model = fsdp_wrapper(model, **model_config.FSDP_CONFIG)
        else:
            model = DINOHead(
                model_config=model_config,
                in_dim=embed_dim,
                num_clusters=[num_clusters],
                bottleneck_dim=bottleneck_dim,
                normalize_last_layer=normalize_last_layer,
            ).cuda()
            model = DistributedDataParallel(model, device_ids=[gpu_id])

        # Print the model
        if gpu_id == 0:
            print(model)

        # Create the optimizer
        param_groups = [
            {
                "params": model.parameters(),
                "lr": 1e-4,
                "weight_decay": 1e-3,
            }
        ]
        optimizer = optim.AdamW(param_groups)

        # Go through several training loops
        losses = []
        for step in range(5):

            # Setup the AMP context if necessary
            context = contextlib.suppress()
            if with_amp:
                context = torch.cuda.amp.autocast()

            # Forward pass
            with context:
                out = model(batch)
                loss = out[0].mean()

            # Backward pass
            if torch_version() >= (1, 7, 0):
                model.zero_grad(set_to_none=True)
            else:
                model.zero_grad()
            loss.backward()
            optimizer.step()

            # Report results and run schedulers
            torch.distributed.all_reduce(loss)
            losses.append(loss.item())
            optimizer.param_groups[0].update(
                {
                    "params": model.parameters(),
                    "lr": 1e-4 + step * 1e-4,
                    "weight_decay": 1e-3 + step * 1e-3,
                }
            )

            # Report memory usage
            if gpu_id == 0:
                print(torch.cuda.max_memory_allocated() // 1e6)

        # Dump the list of losses
        if gpu_id == 0:
            print(losses)
            with open(result_file, "wb") as f:
                pickle.dump(losses, f)

    """
    ---------------------------------------------------------------------------
    Testing ViT VISSL end-to-end implementation
    ---------------------------------------------------------------------------
    """

    @staticmethod
    def _create_dino_pretraining_config(
        with_fsdp: bool,
        with_mixed_precision: bool = False,
        with_normalized_prototypes: bool = True,
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino",
                "config.SEED_VALUE=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        config["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["NUM_LAYERS"] = 1
        if with_fsdp:
            config["MODEL"]["TRUNK"]["NAME"] = "vision_transformer_fsdp"
            config["MODEL"]["HEAD"]["PARAMS"][0][0] = "dino_head_fsdp"
            config.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
            config.MODEL.FSDP_CONFIG.mixed_precision = with_mixed_precision
            config.MODEL.FSDP_CONFIG.fp32_reduce_scatter = with_mixed_precision
            config.MODEL.FSDP_CONFIG.compute_dtype = torch.float32
        config.MODEL.HEAD.PARAMS[0][1][
            "normalize_last_layer"
        ] = with_normalized_prototypes
        return config

    @gpu_test(gpu_count=2)
    def test_vit_fsdp_vs_ddp_convergence(self) -> None:
        with_amp = False
        with with_temp_files(count=2) as file_names:
            self._run_vit_training_loop(
                with_fsdp=True, with_amp=with_amp, output_file_name=file_names[0]
            )
            self._run_vit_training_loop(
                with_fsdp=False, with_amp=with_amp, output_file_name=file_names[1]
            )

            results = []
            for file_name in file_names:
                with open(file_name, "rb") as f:
                    result = pickle.load(f)
                    results.append(result)

            for r0, r1 in zip(results[0], results[1]):
                print(r0, "VS", r1)
            self.assertEqual(results[0], results[1], "DDP vs FSDP")

    @classmethod
    def _run_vit_training_loop(
        cls, with_fsdp: bool, with_amp: bool, output_file_name: str
    ):
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                cls._vit_worker,
                (with_fsdp, with_amp, sync_file, output_file_name),
                nprocs=2,
            )

    @classmethod
    def _vit_worker(
        cls,
        gpu_id: int,
        with_fsdp: bool,
        with_amp: bool,
        sync_file: str,
        result_file: str,
    ):
        init_distributed_on_file(world_size=2, gpu_id=gpu_id, sync_file=sync_file)
        torch.manual_seed(gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()

        # Create the inputs
        batch_size = 8
        batch = torch.randn(size=(batch_size, 3, 224, 224)).cuda()

        # Create the model
        config = cls._create_dino_pretraining_config(with_fsdp=with_fsdp)
        model = build_model(config["MODEL"], config["OPTIMIZER"]).cuda()

        # Build the model with FSDP or DDP
        if with_fsdp:
            model = fsdp_wrapper(model, **config["MODEL"]["FSDP_CONFIG"])
            assert is_valid_fsdp_model(model)
        else:
            model = DistributedDataParallel(model, device_ids=[gpu_id])

        # Print the model
        if gpu_id == 0:
            print(model)

        # Create the optimizer
        param_groups = [
            {
                "params": model.parameters(),
                "lr": 1e-4,
                "weight_decay": 0.0,
            }
        ]
        optimizer = optim.AdamW(param_groups)

        # Go through several training loops
        losses = []
        num_steps = 2
        for step in range(num_steps):

            # Setup the AMP context if necessary
            context = contextlib.suppress()
            if with_amp:
                context = torch.cuda.amp.autocast()

            # Forward pass
            with context:
                out = model(batch)
                out = out[0][0].mean()

            # Backward pass
            if torch_version() >= (1, 7, 0):
                model.zero_grad(set_to_none=True)
            else:
                model.zero_grad()
            out.backward()
            optimizer.step()

            # Report results and run schedulers
            torch.distributed.all_reduce(out)
            losses.append(out.item())
            optimizer.param_groups[0].update(
                {
                    "params": model.parameters(),
                    "lr": 1e-4 + step * 1e-4,
                    "weight_decay": step * 1e-3,
                }
            )

            # Report memory usage
            if gpu_id == 0:
                print(torch.cuda.max_memory_allocated() // 1e6)

        # Dump the list of losses
        if gpu_id == 0:
            print(losses)
            with open(result_file, "wb") as f:
                pickle.dump(losses, f)
