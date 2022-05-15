# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import hashlib
import logging
import os
import re
from enum import auto, Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from classy_vision.generic.util import (
    load_and_broadcast_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from fairscale.nn import FullyShardedDataParallel
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import abspath, create_file_symlink, makedir
from vissl.utils.layer_memory_tracking import null_context


class CheckpointItemType(Enum):
    """
    The different types of checkpoint content:
    - consolidated: a checkpoint containing all the model weights
    - shard: a checkpoint containing the weights of a shard of the model
    - shard_list: a checkpoint listing all shards of a model
    - slice: a checkpoint containing the full weights of a slice of the model
    - slice_list: a checkpoint containing the list of slices for a full model

    Consolidated checkpoints contain all the weights of a model and are
    suitable for models of limited number of parameters.

    For bigger models, we have two different ways to cut the weights:
    - by sharding the model: each shard contains a part of each layer
      and the shard_list contains the path of all the shards
    - by slicing model: each part contains complete parameters weights
      for a sub-part of the model
    """

    consolidated = auto()
    shard = auto()
    shard_list = auto()
    slice = auto()
    slice_list = auto()


class CheckpointWriter:
    """
    Utility class to save checkpoints on the chosen backend
    """

    def __init__(
        self,
        checkpoint_folder: str,
        is_final_train_phase: bool,
        mode: str,
        mode_num: int,
        backend: str,
    ):
        assert backend == "disk", "Only disk BACKEND supported"
        self.checkpoint_folder = checkpoint_folder
        self.is_final_train_phase = is_final_train_phase
        self.mode = mode
        self.mode_num = mode_num

    def save_consolidated_checkpoint(self, content: Dict[str, Any]):
        """
        Save a checkpoint containing the full model weights
        (to be used with DDP on primary rank)
        """

        # Complete the checkpoint with its type
        content["type"] = CheckpointItemType.consolidated.name

        checkpoint_name = self.get_checkpoint_name()
        self._save(name=checkpoint_name, content=content)
        self._create_symbolic_link(checkpoint_name)

    def save_sharded_checkpoint(
        self, content: Dict[str, Any], shard_rank: int, world_size: int
    ):
        """
        Save a checkpoint containing only the model weights of the
        current shard (to be used with FSDP on all ranks)
        """

        # Complete the checkpoint with its type
        content["type"] = CheckpointItemType.shard.name

        # Each worker saves its own shard
        shard_name = self.get_checkpoint_shard_name(shard_rank)
        self._save(name=shard_name, content=content)
        if shard_rank != 0:
            return

        # While the primary worker saves a checkpoint referencing all the shards
        primary_name = self.get_checkpoint_name()
        primary_checkpoint = {
            "type": CheckpointItemType.shard_list.name,
            "shards": [
                self.get_checkpoint_shard_name(rank) for rank in range(world_size)
            ],
        }
        self._save(name=primary_name, content=primary_checkpoint)
        self._create_symbolic_link(primary_name)

    def get_checkpoint_name(self):
        if self.is_final_train_phase:
            return f"model_final_checkpoint_{self.mode}{self.mode_num}.torch"
        return f"model_{self.mode}{self.mode_num}.torch"

    def get_checkpoint_shard_name(self, rank: int):
        if self.is_final_train_phase:
            return (
                f"model_final_checkpoint_{self.mode}{self.mode_num}_shard{rank}.torch"
            )
        return f"model_{self.mode}{self.mode_num}_shard{rank}.torch"

    def _save(self, name: str, content):
        save_checkpoint(
            checkpoint_folder=self.checkpoint_folder,
            state=content,
            checkpoint_file=name,
        )
        logging.info(f"Saved checkpoint: {self.checkpoint_folder}/{name}")

    def _create_symbolic_link(self, checkpoint_name: str):
        """
        Create a "checkpoint.torch" symbolic link that will point to the latest
        checkpoint version.

        It is a particularly useful feature for resuming trainings.
        """
        logging.info("Creating symlink...")
        symlink_dest_file = f"{self.checkpoint_folder}/checkpoint.torch"
        source_file = f"{self.checkpoint_folder}/{checkpoint_name}"
        create_file_symlink(source_file, symlink_dest_file)
        logging.info(f"Created symlink: {symlink_dest_file}")


class CheckpointLoader:
    """
    Utility class to load checkpoints on the chosen backend
    """

    @classmethod
    def init_fsdp_model_from_weights(
        cls,
        model: FullyShardedDataParallel,
        checkpoint: Dict[str, Any],
        weights_path: List[str],
        strict: bool = True,
        head_index: int = -1,
    ):
        """
        Load the weights of the checkpoint to the FSDP model:
        - Take into account the type of checkpoint to decide on how
          to perform the load (sharded or consolidated load)
        - Takes into account the head_index (-1 if trunk else >= 0)
          to find the appropriate weights for the head
        """
        if checkpoint["type"] == CheckpointItemType.slice_list.name:
            # Hack for checkpoints consolidated with the "layers" format
            # instead of the new "classy_state_dict" format: in that case
            # the slices are directly saved under "layers" and do not take
            # into account the 'weights_path' variable
            if "classy_state_dict" not in checkpoint:
                weights = checkpoint["layers"]
            else:
                weights = cls._extract_weights(checkpoint, weights_path, head_index)
            if weights is not None:
                SlicedCheckpointLoader.load_slice_state_dict(
                    model, weights, strict=strict
                )
            else:
                raise ValueError(f"Could not find weights path: {weights_path}")
        elif checkpoint["type"] == CheckpointItemType.consolidated.name:
            weights = cls._extract_weights(checkpoint, weights_path, head_index)
            if weights is not None:
                out = model.load_state_dict(weights, strict=False)
                cls._check_load_state_dict_out(out, strict=strict)
            elif strict:
                raise ValueError(f"Could not find weights path: {weights_path}")
        else:
            weights = cls._extract_weights(checkpoint, weights_path, head_index)
            if weights is not None:
                out = model.load_local_state_dict(weights, strict=False)
                cls._check_load_state_dict_out(out, strict=strict)
            elif strict:
                raise ValueError(f"Could not find weights path: {weights_path}")

    @staticmethod
    def _check_load_state_dict_out(out, strict: bool):
        logging.info(f"Extra layers not loaded: {out.unexpected_keys}")
        if strict and len(out.missing_keys) > 0:
            raise ValueError(f"Could not load keys: {out.missing_keys}")
        elif len(out.missing_keys) > 0:
            logging.info(f"Could not load keys: {out.missing_keys}")

    @classmethod
    def load_and_broadcast_init_weights(cls, checkpoint_path: str, device):
        """
        Load the weights at the provided path, dealing with the
        potential indirection due to the notion of sharded checkpoint
        """
        folder, _ = os.path.split(checkpoint_path)
        return cls.load_and_broadcast_checkpoint(folder, checkpoint_path, device)

    @classmethod
    def load_and_broadcast_checkpoint(
        cls, checkpoint_folder: str, checkpoint_path: str, device
    ) -> Optional[Dict]:
        """
        Load the checkpoint at the provided path, dealing with the
        potential indirection due to the notion of sharded checkpoint
        """
        checkpoint = load_and_broadcast_checkpoint(checkpoint_path, device)
        if checkpoint is None:
            return checkpoint

        cls._update_version(checkpoint)
        if cls._is_shard_aggregator_checkpoint(checkpoint):
            _, global_rank = get_machine_local_and_dist_rank()
            shard_name = checkpoint["shards"][global_rank]
            shard_path = os.path.join(checkpoint_folder, shard_name)
            checkpoint = load_checkpoint(shard_path, device)
        return checkpoint

    @staticmethod
    def _is_shard_aggregator_checkpoint(checkpoint: Dict[str, Any]):
        cp_type = checkpoint.get("type", CheckpointItemType.consolidated.name)
        return cp_type == CheckpointItemType.shard_list.name

    @staticmethod
    def _update_version(checkpoint: Dict[str, Any]):
        # Backward compatibility with old checkpoints saved without types
        checkpoint.setdefault("type", CheckpointItemType.consolidated.name)

    @staticmethod
    def _extract_weights(
        checkpoint: Dict[str, Any], weights_path: List[str], head_index: int = -1
    ):
        weights = checkpoint
        for key in weights_path:
            weights = weights.get(key, None)
            if weights is None:
                return weights

        # If it is the trunk, we already found the correct weights
        if head_index < 0:
            return weights

        # If it is a head, we have two different formats:
        # - either it is a list and we simply have to index with the 'head_index'
        # - or we have a dictionary and so we extract weights with the correct prefix
        if isinstance(weights, list):
            return weights[head_index]
        elif isinstance(weights, dict):
            prefix = CheckpointFormatConverter.to_head_prefix(head_index)
            return {
                k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)
            }
        else:
            return None


class CheckpointFormatConverter:
    """
    Convert a checkpoint from one format to another format more suited
    for evaluation of the model
    """

    @classmethod
    def sharded_to_consolidated_checkpoint(
        cls, input_checkpoint_path: str, output_checkpoint_path: str
    ):
        """
        Given a path to a sharded checkpoint, create a consolidated checkpoint
        in which the weights of the trunk are stitched back together

        This function does not copy the optimizer state nor the head
        state as these states are not used for evaluation
        """
        weights, metadata, heads_weights, heads_metadata = cls._read_shards(
            input_checkpoint_path
        )
        full_trunk_weights = cls._consolidate_shards(weights, metadata)
        if heads_metadata:
            full_heads_weights = {
                cls.to_head_prefix(head_index) + param_name: param
                for head_index, head_weights in heads_weights.items()
                for param_name, param in cls._consolidate_shards(
                    head_weights, heads_metadata[head_index]
                ).items()
            }
        else:
            full_heads_weights = {}
        consolidated_checkpoint = {
            "type": CheckpointItemType.consolidated.name,
            "classy_state_dict": {
                "base_model": {
                    "model": {"trunk": full_trunk_weights, "heads": full_heads_weights}
                }
            },
        }
        logging.info(f"Saving consolidated checkpoint at: {output_checkpoint_path}")
        with g_pathmgr.open(output_checkpoint_path, "wb") as f:
            torch.save(consolidated_checkpoint, f)
        logging.info(f"Done! Checkpoint available at: {output_checkpoint_path}")

    @classmethod
    def to_sliced_checkpoint(
        cls, input_checkpoint_path: str, output_checkpoint_path: str
    ):
        """
        Given a path to either a consolidated or sharded checkpoint, create a sliced
        checkpoint in which the weights of the trunk are saved in separate files
        """
        with g_pathmgr.open(input_checkpoint_path, "rb") as f:
            cp = torch.load(f, map_location="cpu")

        cp_type = cp.get("type", CheckpointItemType.consolidated.name)
        if cp_type == CheckpointItemType.consolidated.name:
            logging.info(
                f"Start slicing consolidated checkpoint {input_checkpoint_path}..."
            )
            cls.consolidated_to_sliced_checkpoint(
                input_checkpoint_path, output_checkpoint_path, pre_loaded_checkpoint=cp
            )
        else:
            logging.info(f"Start slicing sharded checkpoint {input_checkpoint_path}...")
            cls.sharded_to_sliced_checkpoint(
                input_checkpoint_path, output_checkpoint_path
            )

    @classmethod
    def consolidated_to_sliced_checkpoint(
        cls,
        input_checkpoint_path: str,
        output_checkpoint_path: str,
        pre_loaded_checkpoint: Optional[dict] = None,
    ):
        """
        Given a path to a consolidated checkpoint, create a sliced checkpoint
        in which the weights of the trunk are saved in separate files

        This is particularly useful for FSDP models, when you have a consolidated
        checkpoint that cannot be loaded at once because of limiting GPU memory
        but can be loaded incrementally.
        """
        if pre_loaded_checkpoint is not None:
            # If checkpoint has already been loaded to check its type,
            # use the loaded checkpoint instead of loading it again
            conso_cp = pre_loaded_checkpoint
        else:
            with g_pathmgr.open(input_checkpoint_path, "rb") as f:
                conso_cp = torch.load(f, map_location="cpu")

        assert conso_cp["type"] == CheckpointItemType.consolidated.name
        trunk_weights = conso_cp["classy_state_dict"]["base_model"]["model"]["trunk"]
        head_weights = conso_cp["classy_state_dict"]["base_model"]["model"]["heads"]

        saved_trunk_parameters = {}
        for param_path, param in trunk_weights.items():
            file_path = SlicedCheckpointLoader.save_slice(
                output_checkpoint_path, param_path, param
            )
            saved_trunk_parameters[param_path] = file_path

        saved_head_parameters = {}
        for param_path, param in head_weights.items():
            file_path = SlicedCheckpointLoader.save_slice(
                output_checkpoint_path, param_path, param
            )
            saved_head_parameters[param_path] = file_path

        cls._save_slice_list(
            saved_trunk_parameters, saved_head_parameters, output_checkpoint_path
        )

    @classmethod
    def sharded_to_sliced_checkpoint(
        cls, input_checkpoint_path: str, output_checkpoint_path: str
    ):
        """
        Given a path to a sharded checkpoint, create a sliced checkpoint
        in which the weights of the trunk are stitched back together
        before saving them weight by weights in separate files

        This function does not copy the optimizer state nor the head
        state as these states are not used for evaluation
        """
        trunk_weights, trunk_metadata, heads_weights, heads_metadata = cls._read_shards(
            input_checkpoint_path
        )

        saved_trunk_parameters = {}
        trunk_weights = cls._consolidate_shards(trunk_weights, trunk_metadata)
        for param_path, param in trunk_weights.items():
            file_path = SlicedCheckpointLoader.save_slice(
                output_checkpoint_path, param_path, param
            )
            saved_trunk_parameters[param_path] = file_path

        saved_head_parameters = {}
        for head_index, head_weights in heads_weights.items():
            conso_weights = cls._consolidate_shards(
                head_weights, heads_metadata[head_index]
            )
            for param_path, param in conso_weights.items():
                full_param_path = cls.to_head_prefix(head_index) + param_path
                file_path = SlicedCheckpointLoader.save_slice(
                    output_checkpoint_path, full_param_path, param
                )
                saved_head_parameters[full_param_path] = file_path

        cls._save_slice_list(
            saved_trunk_parameters, saved_head_parameters, output_checkpoint_path
        )

    @classmethod
    def _save_slice_list(
        cls,
        saved_trunk_parameters: Dict[str, str],
        saved_head_parameters: Dict[str, str],
        output_checkpoint_path: str,
    ):
        checkpoint_list = {
            "type": CheckpointItemType.slice_list.name,
            "classy_state_dict": {
                "base_model": {
                    "model": {
                        "trunk": saved_trunk_parameters,
                        "heads": saved_head_parameters,
                    }
                }
            },
        }

        logging.info(f"Saving sliced checkpoint at: {output_checkpoint_path}")
        with g_pathmgr.open(output_checkpoint_path, "wb") as f:
            torch.save(checkpoint_list, f)
        logging.info(f"Done! Checkpoint available at: {output_checkpoint_path}")

    @staticmethod
    def to_head_prefix(head_index: int) -> str:
        return str(head_index) + "."

    @classmethod
    def _read_shards(cls, input_checkpoint_path: str, device="cpu"):
        logging.info(f"Reading sharded checkpoint from: {input_checkpoint_path}")
        with g_pathmgr.open(input_checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location=device)

        assert checkpoint["type"] == CheckpointItemType.shard_list.name
        trunk_weights, trunk_metadata = [], []
        head_weights, head_metadata = {}, {}

        for shard_path in checkpoint["shards"]:
            if not os.path.isabs(shard_path):
                checkpoint_folder = os.path.split(input_checkpoint_path)[0]
                shard_path = os.path.join(checkpoint_folder, shard_path)

            with g_pathmgr.open(shard_path, "rb") as f:
                shard_content = torch.load(f, map_location=device)

            # Consolidate the trunk weights based on the meta-data
            shard_data = shard_content["classy_state_dict"]["base_model"]["model"]
            shard_meta = shard_content["classy_state_dict"]["base_model"]["meta"]
            trunk_weights.append(shard_data["trunk"])
            trunk_metadata.append(shard_meta["trunk"])

            # In case there are meta-data about the head, consolidate the head as well
            if "heads" in shard_meta:
                assert (
                    "heads" in shard_data
                ), f"Expected head weights in checkpoint: {shard_path}"
                heads_data = shard_data["heads"]
                heads_meta = shard_meta["heads"]
                for i, (head_data, head_meta) in enumerate(zip(heads_data, heads_meta)):
                    head_weights.setdefault(i, []).append(head_data)
                    head_metadata.setdefault(i, []).append(head_meta)

        return trunk_weights, trunk_metadata, head_weights, head_metadata

    @classmethod
    def _consolidate_shards(
        cls, weights: List[Dict[str, torch.Tensor]], metadata: List[Dict[str, Any]]
    ):
        logging.info("Consolidating shards...")
        return FullyShardedDataParallel.consolidate_shard_weights(weights, metadata)


class SlicedCheckpointLoader:
    """
    Save a checkpoint by scanning the parameters one by one and saving
    them on the fly, summoning the minimum number of FSDP parameters
    possible along the way.
    """

    @classmethod
    def save_slice(cls, checkpoint_path: str, param_path: str, param) -> str:
        """
        Save a slice of the model: a parameter and its associated weights
        - create a folder in which the slice will live
        - save the slice in this folder, with a unique name
        - return the created file name
        """
        checkpoint_sub_folder = os.path.splitext(checkpoint_path)[0] + "_layers"
        makedir(checkpoint_sub_folder)
        hash_name = hashlib.sha1(param_path.encode()).hexdigest()
        file_path = os.path.join(checkpoint_sub_folder, f"{hash_name}.torch")
        file_path = abspath(file_path)
        checkpoint_slice = {"type": CheckpointItemType.slice.name, "weight": param}
        with g_pathmgr.open(file_path, "wb") as f:
            torch.save(checkpoint_slice, f)
        return file_path

    @classmethod
    def load_slice_state_dict(
        cls,
        model: FullyShardedDataParallel,
        slice_state_dict: Dict[str, str],
        strict: bool = True,
    ):
        """
        Given a static dict associating parameter names to the path of the
        file containing the corresponding weights (for lazy loading),
        initialize the weights of the model layer by layer, summoning the
        parameters on the fly to avoid OOM
        """
        for path, module in cls._recursive_visit(model):
            for param_path, param in module.named_parameters(
                prefix=path, recurse=False
            ):
                cls._init_weight_from_slice(
                    param_path, param.data, slice_state_dict, strict=strict
                )
            for buffer_path, buffer in module.named_buffers(prefix=path, recurse=False):
                cls._init_weight_from_slice(
                    buffer_path, buffer.data, slice_state_dict, strict=strict
                )

    @classmethod
    def _init_weight_from_slice(
        cls,
        weight_path: str,
        weight: torch.Tensor,
        slice_state_dict: Dict[str, str],
        strict: bool = True,
    ):
        weight_path = cls._clean_path(weight_path)
        file_name = slice_state_dict.get(weight_path, None)
        if file_name is None:
            message = f"Could not find weights: {weight_path}"
            logging.info(message)
            if strict:
                raise ValueError(
                    f"Could not find weights: {weight_path} among:\n{slice_state_dict.keys()}"
                )
            return

        logging.info(f"Loading weights: {weight_path}")
        with g_pathmgr.open(file_name, "rb") as f:
            layer_checkpoint = torch.load(f)

        assert layer_checkpoint["type"] == CheckpointItemType.slice.name
        weight.copy_(layer_checkpoint["weight"])
        logging.info(f"Loaded parameters '{weight_path}' from: {file_name}")

    @classmethod
    def _recursive_visit(cls, model: FullyShardedDataParallel):
        """
        Visit a FSDP model, summoning parameters on the fly
        and releasing them as soon as they are not needed

        This replicates the summoning of parameters as done
        through the forward pass of a FSDP model
        """

        def visit(path, module):
            context = null_context()
            if isinstance(module, FullyShardedDataParallel):
                context = cls._summon_params(module)

            with context:
                yield path, module
                for name, child in module._modules.items():
                    next_path = path + "." + name if path else name
                    yield from visit(next_path, child)

        yield from visit("", model)

    @staticmethod
    @contextlib.contextmanager
    def _summon_params(module):
        with module.summon_full_params(recurse=False):
            yield

    @staticmethod
    def _clean_path(param_path: str):
        fsdp_names = {"_fsdp_wrapped_module", "_fpw_module"}
        return ".".join(
            [split for split in param_path.split(".") if split not in fsdp_names]
        )


def is_training_finished(cfg: AttrDict, checkpoint_folder: str):
    """
    Given the checkpoint folder, we check that there's not already a final checkpoint
    If the final checkpoint exists but the user wants to override the final checkpoint
    then we mark training as not finished.

    Args:
        cfg (AttrDict): input config file specified by user and parsed by vissl
        checkpoint_folder (str): the directory where the checkpoints exist

    Returns:
        boolean whether training is finished or not.
    """
    if not cfg["CHECKPOINT"]["OVERWRITE_EXISTING"] and has_final_checkpoint(
        checkpoint_folder
    ):
        return True


def get_checkpoint_folder(config: AttrDict):
    """
    Check, create and return the checkpoint folder. User can specify their own
    checkpoint directory otherwise the default "." is used.

    Optionally, for training that involves more than 1 machine, we allow to append
    the distributed run id which helps to uniquely identify the training. This is
    completely optional and user can se APPEND_DISTR_RUN_ID=true for this.
    """
    odir = config.CHECKPOINT.DIR
    if config.DISTRIBUTED.NUM_NODES > 1 and config.CHECKPOINT.APPEND_DISTR_RUN_ID:
        odir = f"{odir}/{config.DISTRIBUTED.RUN_ID}"

    makedir(odir)
    assert g_pathmgr.exists(
        config.CHECKPOINT.DIR
    ), f"Please specify config.CHECKPOINT.DIR parameter. Invalid: {config.CHECKPOINT.DIR}"
    return odir


def is_checkpoint_phase(
    mode_num: int,
    mode_frequency: int,
    train_phase_idx: int,
    num_train_phases: int,
    mode: str,
):
    """
    Determines if a checkpoint should be saved on current epoch. If epoch=1, then
    we check whether to save at current iteration or not.

    Args:
        mode (str): what model we are checkpointing models at - every few iterations or
                    at the end of every phase/epoch. The mode is encoded in the checkpoint
                    filename.
        mode_num (int): what is the current iteration or epoch number that we are trying to
                        checkpoint at.
        mode_frequency (int): checkpoint frequency - every N iterations or every N epochs/phase
        train_phase_idx (int): the current training phase we are in. Starts from 0
        num_train_phases (int): total number of training phases. Usually the same as num_epochs.

    Returns:
        checkpointing_phase (bool): whether the model should be checkpointed or not
    """
    if mode == "iteration":
        checkpointing_phase = (mode_num % mode_frequency) == 0
    elif mode == "phase":
        checkpointing_phase = (mode_num % mode_frequency) == 0 or train_phase_idx == (
            num_train_phases - 1
        )
    return checkpointing_phase


def has_checkpoint(checkpoint_folder: str, skip_final: bool = False):
    """
    Check whether there are any checkpoints at all in the checkpoint folder.

    Args:
        checkpoint_folder (str): path to the checkpoint folder
        skip_final (bool): if the checkpoint with `model_final_` prefix exist, whether
                           to skip it and train.

    Returns:
        checkpoint_exists (bool): whether checkpoint exists or not
    """
    checkpointed_files = g_pathmgr.ls(checkpoint_folder)
    checkpoint_exists = False
    for f in checkpointed_files:
        if f.endswith(".torch") and ("model_final" not in f or not skip_final):
            checkpoint_exists = True
            break
    return checkpoint_exists


def has_final_checkpoint(
    checkpoint_folder: str, final_checkpoint_pattern: str = "model_final"
):
    """
    Check whether the final checkpoint exists in the checkpoint folder. The
    final checkpoint is recognized by the prefix "model_final_" in VISSL.

    Args:
        checkpoint_folder (str): path to the checkpoint folder.
        final_checkpoint_pattern (str): what prefix is used to save the final checkpoint.

    Returns:
        has_final_checkpoint: whether the final checkpoint exists or not
    """
    checkpointed_files = g_pathmgr.ls(checkpoint_folder)
    torch_files = filter(lambda x: x.endswith(".torch"), checkpointed_files)
    final_files = filter(lambda x: final_checkpoint_pattern in x, torch_files)
    return len(list(final_files)) > 0


def get_checkpoint_resume_files(
    checkpoint_folder: str,
    config: AttrDict,
    skip_final: bool = False,
    latest_checkpoint_resume_num: int = 1,
):
    """
    Get the checkpoint file from which the model should be resumed. We look at all
    the checkpoints in the checkpoint_folder and if the final model checkpoint exists
    (starts with `model_final_`) and not overriding it, then return the final
    checkpoint. Otherwise find the latest checkpoint.

    Args:
        checkpoint_folder (str): path to the checkpoint folder.
        config (AttrDict): root config
        skip_final (bool): whether the final model checkpoint should be skipped or not
        latest_checkpoint_resume_num (int): what Nth latest checkpoint to resume from.
                   Sometimes the latest checkpoints could be corrupt so this option
                   helps to resume from instead a few checkpoints before the last checkpoint.
    """
    all_files = g_pathmgr.ls(checkpoint_folder)
    all_iters = []
    replace_prefix = "model_phase"
    # if we checkpoint at iterations too, we start from an iteration checkpoint
    # since that's latest than the phase end checkpoint. Sometimes, it's also
    # possible that there is no phase.
    if config.CHECKPOINT.CHECKPOINT_ITER_FREQUENCY > 0:
        replace_prefix = "model_iteration"

    for f in all_files:
        # if we have the finished training, we pick the finished training file
        # the checkpoint is saved as "model_final_checkpoint". Otherwise, we pick
        # the latest phase checkpoint
        if "model_final" in f and not skip_final:
            return f
        if replace_prefix in f:
            iter_num = f.replace(".torch", "").replace(replace_prefix, "")
            if iter_num.isdigit():
                all_iters.append(int(iter_num))

    # make sure the checkpoint resume number is in bounds
    checkpoint_resume_num = max(0, latest_checkpoint_resume_num - 1)
    # len(all_iters) - 1 is the last index, checkpoint_resume_num can't be beyond that.
    checkpoint_resume_num = min(len(all_iters) - 1, checkpoint_resume_num)
    logging.info(f"checkpoint_resume_num: {checkpoint_resume_num}")
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[checkpoint_resume_num])
        filename = f"{replace_prefix}{last_iter}.torch"
        return filename
    else:
        return None


def get_resume_checkpoint(cfg: AttrDict, checkpoint_folder: str):
    """
    Return the checkpoint from which to resume training. If no checkpoint found,
    return None. Resuming training is optional and user can set AUTO_RESUME=false
    to not resume the training.

    If we want to overwrite the existing final checkpoint, we ignore the final
    checkpoint and return the previous checkpoints if exist.
    """
    # we check whether there's a checkpoint that already exists
    checkpoint_path = None
    # if we are overwriting the existing checkpoint, then skip_final=true in
    # `has_checkpoint` call
    checkpoints_exists = has_checkpoint(
        checkpoint_folder, skip_final=cfg["CHECKPOINT"]["OVERWRITE_EXISTING"]
    )
    if checkpoints_exists and cfg["CHECKPOINT"]["AUTO_RESUME"]:
        checkpoint_file = get_checkpoint_resume_files(
            checkpoint_folder,
            cfg,
            skip_final=cfg["CHECKPOINT"]["OVERWRITE_EXISTING"],
            latest_checkpoint_resume_num=cfg["CHECKPOINT"][
                "LATEST_CHECKPOINT_RESUME_FILE_NUM"
            ],
        )

        checkpoint_path = f"{checkpoint_folder}/{checkpoint_file}"
        logging.info(f"Resume from file: {checkpoint_path}")
    return checkpoint_path


def print_state_dict_shapes(state_dict: Dict[str, Any]):
    """
    For the given model state dictionary, print the name and shape
    of each parameter tensor in the model state. Helps debugging.

    Args:
        state_dict (Dict[str, Any]): model state dictionary
    """
    logging.info("Model state_dict:")
    for param_tensor in state_dict.keys():
        logging.info(f"{param_tensor}:\t{state_dict[param_tensor].size()}")


def print_loaded_dict_info(
    model_state_dict: Dict[str, Any],
    state_dict: Dict[str, Any],
    skip_layers: List[str],
    model_config: AttrDict,
):
    """
    Print what layers were loaded, what layers were ignored/skipped/not found
    when initializing a model from a specified model params file.
    """
    extra_layers = []
    max_len_model = max(len(key) for key in model_state_dict.keys())
    # go through the model layers and print what layer is loaded/not loaded/skipped
    for layername in model_state_dict.keys():
        if len(skip_layers) > 0 and any(item in layername for item in skip_layers):
            logging.info(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            if ("heads" not in layername) or should_init_head_weights(model_config):
                logging.info(
                    f"Loaded: {layername: <{max_len_model}} of "
                    f"shape: {model_state_dict[layername].size()} from checkpoint"
                )
            else:
                logging.info(f"Ignored layer:\t{layername}")
        else:
            logging.info(f"Not found:\t\t{layername}, not initialized")

    # go through the checkpoint state_dict and print what extra layers exist in checkpoint
    for layername in state_dict.keys():
        if layername not in model_state_dict:
            extra_layers.append(layername)
    logging.info(f"Extra layers not loaded from checkpoint: {extra_layers}")


def replace_module_prefix(
    state_dict: Dict[str, Any], prefix: str, replace_with: str = ""
):
    """
    Remove prefixes in a state_dict needed when loading models that are not VISSL
    trained models.

    Specify the prefix in the keys that should be removed.
    """
    state_dict = {
        (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def append_module_prefix(state_dict: Dict[str, Any], prefix: str):
    """
    Append prefixes in a state_dict needed when loading models that are not VISSL
    trained models.

    In order to load the model (if not trained with VISSL) with VISSL, there are 2 scenarios:
        1. If you are interested in evaluating the model features and freeze the trunk.
           Set APPEND_PREFIX="trunk.base_model." This assumes that your model is compatible
           with the VISSL trunks. The VISSL trunks start with "_feature_blocks." prefix. If
           your model doesn't have these prefix you can append them. For example:
           For TorchVision ResNet trunk, set APPEND_PREFIX="trunk.base_model._feature_blocks."
        2. where you want to load the model simply and finetune the full model.
           Set APPEND_PREFIX="trunk."
           This assumes that your model is compatible with the VISSL trunks. The VISSL
           trunks start with "_feature_blocks." prefix. If your model doesn't have these
           prefix you can append them.
           For TorchVision ResNet trunk, set APPEND_PREFIX="trunk._feature_blocks."
     NOTE: the prefix is appended to all the layers in the model
    """
    state_dict = {f"{prefix}{key}": val for (key, val) in state_dict.items()}
    return state_dict


def check_model_compatibility(state_dict: Dict[str, Any]):
    """
    Given a state_dict, perform basic check to verify if anything in the state_dict
    can be loaded to a VISSL model ('trunk' or 'head'):

    The goal of this function is not to exclude a checkpoint if we find something
    that is not compatible (it is okay not to load everything in a state_dict) but
    instead to catch gross mistakes where the state_dict does not contain any useful
    information for a VISSL model. In such cases, we raise exception.

    Args:
        state_dict (Dict[str, Any]): state dict that should be checked for compatibility
    """
    useful_prefixes = {"trunk.", "heads."}
    for layer_name in state_dict.keys():
        if any(layer_name.startswith(prefix) for prefix in useful_prefixes):
            return

    raise Exception(
        "Model provided in config.MODEL.WEIGHTS_INIT.PARAMS_FILE is not compatible "
        "with VISSL. Please set config.MODEL.WEIGHTS_INIT.APPEND_PREFIX and "
        "config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX for making model compatible. "
        f"Expected prefixes: {useful_prefixes}."
    )


def is_feature_extractor_state_dict(state_dict: Dict[str, Any]):
    """
    We check if the trunk state dict is already a feature extractor state dict.
    If it is, return True otherwise return False.
    """
    contains_prefix = [key.startswith("base_model.") for (key, _) in state_dict.items()]
    return np.all(contains_prefix)


def adapt_to_feature_extractor_config(
    config: AttrDict, state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adapt a state dictionary to be compatible with a feature extractor configuration
    by replacing the "trunk." by "trunk.base_model."
    """
    from vissl.models import is_feature_extractor_model

    if not is_feature_extractor_model(config.MODEL):
        return state_dict

    return {k.replace("trunk.", "trunk.base_model."): v for k, v in state_dict.items()}


def get_checkpoint_model_state_dict(config: AttrDict, state_dict: Dict[str, Any]):
    """
    Given a specified pre-trained VISSL model (composed of head and trunk),
    we get the state_dict that can be loaded by appending prefixes to model and trunk.

    Args:
        config (AttrDict): full config file
        state_dict (Dict): raw state_dict loaded from the checkpoint or weights file

    Returns:
        state_dict (Dict): vissl state_dict with layer names matching compatible with
                           vissl model. Hence this state_dict can be loaded directly.
    """
    from vissl.models import is_feature_extractor_model

    classy_state_dict = state_dict["base_model"]["model"]
    trunk_append_prefix, heads_append_prefix = "trunk.", "heads."
    # if the model is being loaded for feature extraction, we check
    # that the model is not already a feature extractor trunk. If
    # not, we add the appropriate prefix to append.
    if is_feature_extractor_model(config.MODEL) and not is_feature_extractor_state_dict(
        classy_state_dict["trunk"]
    ):
        trunk_append_prefix = "trunk.base_model."
    elif not is_feature_extractor_model(config.MODEL):
        # Getting rid of the feature extractor prefix if we do not need it
        classy_state_dict["trunk"] = replace_module_prefix(
            classy_state_dict["trunk"], "base_model.", ""
        )

    trunk_state_dict = append_module_prefix(
        classy_state_dict["trunk"], trunk_append_prefix
    )
    heads_state_dict = append_module_prefix(
        classy_state_dict["heads"], heads_append_prefix
    )
    state_dict = {}
    state_dict.update(trunk_state_dict)
    state_dict.update(heads_state_dict)
    return state_dict


def should_init_head_weights(model_config: AttrDict) -> bool:
    """
    Indicates whether or not we should load the weights of the head
    based on the evaluation mode and evaluation settings
    """
    # Fine-tuning
    if not model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON:
        return True

    # Extraction of features or label prediction at head
    return (
        model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
        and model_config.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD
    )


def init_model_from_consolidated_weights(
    config: AttrDict,
    model,
    state_dict: Dict[str, Any],
    state_dict_key_name: Union[str, List[str]],
    skip_layers: List[str],
    replace_prefix=None,
    append_prefix=None,
    strict: bool = False,
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the feature evaluation process or when we want to evaluate a model on
    a range of tasks.

    Args:
        config (AttrDict): config file
        model (object): instance of base_ssl_model
        state_dict (Dict): torch.load() of user provided params file path.
        state_dict_key_name (string | list): key name containing the model state dict
        skip_layers (List(string)): layer names with this key are not copied
        replace_prefix (string): remove these prefixes from the layer names (executed first)
        append_prefix (string): append the prefix to the layer names
                                (executed after replace_prefix)
        strict (bool): whether or not to raise an error in case layers are not initialized

    Returns:
        model (object): the model initialized from the weights file
    """
    # whether it's a model from somewhere else or a model from this codebase, load the
    # state_dict
    invalid_key_message = f"Unknown state dict key: {state_dict_key_name}"
    if isinstance(state_dict_key_name, list):
        for key in state_dict_key_name:
            assert key in state_dict.keys(), invalid_key_message
            state_dict = state_dict[key]
    elif state_dict_key_name and len(state_dict_key_name) > 0:
        assert state_dict_key_name in state_dict.keys(), invalid_key_message
        state_dict = state_dict[state_dict_key_name]

    if state_dict_key_name == "classy_state_dict":
        # get the appropriate model_state_dict so that the model can load. We automatically
        # take care of appending prefixes, suffixes etc to match the layer names.
        state_dict = get_checkpoint_model_state_dict(config, state_dict)
    else:
        # make any corrections to the layer names to load checkpoint successfully
        if replace_prefix:
            state_dict = replace_module_prefix(state_dict, replace_prefix)
        if append_prefix:
            state_dict = append_module_prefix(state_dict, append_prefix)
        check_model_compatibility(state_dict)
        state_dict = adapt_to_feature_extractor_config(config, state_dict)

    # load the checkpoint now
    all_layers = model.state_dict()
    missing_layers = []

    local_rank, _ = get_machine_local_and_dist_rank()
    max_len_model = max(len(key) for key in all_layers.keys())
    for layername in all_layers.keys():

        # Ignore layers in "skip_layers"
        if len(skip_layers) > 0 and any(item in layername for item in skip_layers):
            if local_rank == 0:
                logging.info(f"Ignored layer:\t{layername}")
            continue

        # Otherwise initialize the layer
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            # if we are initializing the heads and the feature eval mode is on, we check
            # if we are evaluating the heads as well or not. If not, we don't initialize
            # the heads. Otherwise we initialize the heads.
            if not ("heads" in layername) or (
                "heads" in layername and should_init_head_weights(config.MODEL)
            ):
                # Accommodate changing position embeddings. Fine-tuning at a
                # different resolution than that which a model was pretrained
                # at requires interpolating the learned position embeddings.
                if "pos_embedding" in layername:
                    param = interpolate_position_embeddings(
                        model, all_layers[layername], param
                    )
                if (
                    "heads" in layername
                    and not config.MODEL.FEATURE_EVAL_SETTINGS.ASSERT_HEAD_LAYER_SHAPE_INIT
                ):
                    if local_rank == 0:
                        logging.info(
                            f"Ignore shape check: {layername} "
                            f"checkpoint: {param.shape}, model: {all_layers[layername].shape}"
                        )
                    if all_layers[layername].shape == param.shape:
                        all_layers[layername].copy_(param)
                        if local_rank == 0:
                            logging.info(
                                f"Loaded: {layername: <{max_len_model}} of "
                                f"shape: {all_layers[layername].size()} from checkpoint"
                            )
                else:
                    assert all_layers[layername].shape == param.shape, (
                        f"{layername} have different shapes: "
                        f"checkpoint: {param.shape}, model: {all_layers[layername].shape}"
                    )
                    all_layers[layername].copy_(param)
                    if local_rank == 0:
                        logging.info(
                            f"Loaded: {layername: <{max_len_model}} of "
                            f"shape: {all_layers[layername].size()} from checkpoint"
                        )

            # In case the layer is ignored by settings
            else:
                if local_rank == 0:
                    logging.info(f"Ignored layer:\t{layername}")

        # In case the layer cannot be found
        else:
            missing_layers.append(layername)
            if local_rank == 0:
                logging.info(f"Not found:\t\t{layername}, not initialized")

    # Raise an error if some layers are not initialized
    if strict and len(missing_layers) > 0:
        raise ValueError(f"Layers not initialized: {missing_layers}")

    if local_rank == 0:
        # go through the checkpoint state_dict and print what extra layers exist in checkpoint
        extra_layers = [
            layer_name
            for layer_name in state_dict.keys()
            if layer_name not in all_layers
        ]
        logging.info(f"Extra layers not loaded from checkpoint: {extra_layers}")

    ####################### DEBUG ############################
    # print_state_dict_shapes(model.state_dict())
    return model


def interpolate_position_embeddings(model, layer, param):
    """
    Fine-tuning at a different resolution than that which a model was
    pretrained at requires interpolating the learned position embeddings.
    """
    if (
        hasattr(model.trunk, "interpolate_position_embedding")
        and layer.shape != param.shape
    ):
        interp = model.trunk.interpolate_position_embedding
        if callable(interp):
            try:
                param = interp(param)
            except BaseException:
                raise RuntimeError("Unable to interpolate position embeddings")
    return param


class DINOCheckpointUtils:
    """
    Checkpoint utilities to extract the teacher of DINO
    into a standard VISSL checkpoint
    """

    @staticmethod
    def remove_prefix(key: str, prefixes: List[str]):
        """
        Remove one of the prefixes provided as parameter
        """
        for prefix in prefixes:
            if key.startswith(prefix):
                return key.replace(prefix, "")
        raise ValueError(f"Expected one prefix to be removed among {prefixes}")

    @classmethod
    def extract_teacher_from_consolidated_checkpoint(
        cls, input_cp: dict, output_path: str
    ):
        output_folder = os.path.split(output_path)[0]
        makedir(output_folder)

        loss_cp = input_cp["classy_state_dict"]["loss"]
        trunk_weights, heads_weights = {}, {}
        for k, v in loss_cp.items():
            if "trunk" in k:
                k = cls.remove_prefix(
                    k, ["momentum_teacher.module.trunk.", "momentum_teacher.trunk."]
                )
                trunk_weights[k] = v
            elif "heads" in k:
                k = cls.remove_prefix(
                    k, ["momentum_teacher.module.heads.", "momentum_teacher.heads."]
                )
                heads_weights[k] = v
        output_cp = {
            "type": CheckpointItemType.consolidated.name,
            "classy_state_dict": {
                "base_model": {
                    "model": {"trunk": trunk_weights, "heads": heads_weights}
                }
            },
        }
        with g_pathmgr.open(output_path, "wb") as f:
            torch.save(output_cp, f)

    @classmethod
    def extract_teacher_from_sharded_checkpoint(
        cls, input_checkpoint_path: str, output_checkpoint_path: str
    ):
        """
        For sharded checkpoint, we extract the teacher shards from each shard and save a new
        sharded checkpoint where the shards weights contain the teacher shard weights
        """
        with g_pathmgr.open(input_checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
            assert checkpoint["type"] == CheckpointItemType.shard_list.name

        input_folder = os.path.split(input_checkpoint_path)[0]
        output_folder = os.path.split(output_checkpoint_path)[0]
        makedir(output_folder)

        output_shard_list = []
        extract_shard_id = re.compile(r".*_shard([0-9]*)$")
        for input_shard_path in checkpoint["shards"]:
            input_shard_name = os.path.splitext(input_shard_path)[0]
            print(input_shard_name)
            match = extract_shard_id.match(input_shard_name)
            shard_id = match.group(1)

            if not os.path.isabs(input_shard_path):
                input_shard_path = os.path.join(input_folder, input_shard_path)

            with g_pathmgr.open(input_shard_path, "rb") as f:
                shard_content = torch.load(f, map_location="cpu")

            trunk_weights, heads_weights = {}, {}
            for name, value in shard_content["classy_state_dict"]["loss"][
                "teacher"
            ].items():
                if name.startswith("trunk."):
                    trunk_weights[name.replace("trunk.", "")] = value
                elif name.startswith("trunk"):
                    trunk_weights[name.replace("trunk", "")] = value
                elif name.startswith("heads."):
                    trunk_weights[name.replace("heads.", "")] = value
                else:
                    raise ValueError(name)

            shard_meta = shard_content["classy_state_dict"]["loss"]["teacher_meta"]
            shard_param_meta = shard_meta["param_metadata"]
            shard_buffer_names = shard_meta["buffer_names"]

            trunk_meta = {
                "param_metadata": [
                    cls._remove_fsdp_path_prefix(m, "trunk")
                    for m in shard_param_meta
                    if m["fsdp_path"].startswith("trunk")
                ],
                "buffer_names": [
                    n.replace("trunk.", "").replace("trunk", "")
                    for n in shard_buffer_names
                    if n.startswith("trunk")
                ],
            }

            # TODO - fix heads_meta - it should be a list
            # heads_meta = {
            #     "param_metadata": [
            #         cls._remove_fsdp_path_prefix(m, "heads.")
            #         for m in shard_param_meta
            #         if m["fsdp_path"].startswith("heads.")
            #     ],
            #     "buffer_names": [
            #         n.replace("heads.", "")
            #         for n in shard_buffer_names
            #         if n.startswith("heads.")
            #     ],
            # }

            output_cp = {
                "type": CheckpointItemType.shard_list.name,
                "classy_state_dict": {
                    "base_model": {
                        "model": {"trunk": trunk_weights, "heads": heads_weights},
                        # TODO - fix heads_meta - it should be a list
                        "meta": {"trunk": trunk_meta, "heads": []},
                    }
                },
            }

            # Save the shard and record its name
            output_shard_name = os.path.splitext(output_checkpoint_path)[0]
            output_shard_name = output_shard_name + f"_shard{shard_id}.torch"
            output_shard_path = os.path.join(output_folder, output_shard_name)
            output_shard_list.append(output_shard_path)
            with g_pathmgr.open(output_shard_path, "wb") as f:
                torch.save(output_cp, f)

        # Save the shard list checkpoint
        with g_pathmgr.open(output_checkpoint_path, "wb") as f:
            output_shard_list_cp = {
                "type": CheckpointItemType.shard_list.name,
                "shards": output_shard_list,
            }
            torch.save(output_shard_list_cp, f)

    @staticmethod
    def _remove_fsdp_path_prefix(fsdp_meta_data, prefix: str):
        fsdp_path = fsdp_meta_data["fsdp_path"].replace(prefix, "")
        if fsdp_path.startswith("."):
            fsdp_path = fsdp_path[1:]
        fsdp_meta_data["fsdp_path"] = fsdp_path
        return fsdp_meta_data
