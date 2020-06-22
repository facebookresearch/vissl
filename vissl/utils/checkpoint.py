#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import tempfile
import traceback
from shutil import copy2, move

import torch
from classy_vision.generic.util import load_checkpoint
from vissl.utils.io import makedir


def is_training_finished(cfg, checkpoint_folder):
    # given the checkpoint folder,
    # - we check that there's not already a final checkpoint
    if not cfg["CHECKPOINT"]["OVERWRITE_EXISTING"] and has_final_checkpoint(
        checkpoint_folder
    ):
        return True


def get_checkpoint_folder(config):
    odir = None
    if config.CHECKPOINT.DIR:
        odir = os.path.abspath(config.CHECKPOINT.DIR)
    else:
        raise Exception("No config.CHECKPOINT.DIR specified.")
    if config.DISTRIBUTED.NUM_NODES > 1 and config.CHECKPOINT.APPEND_DISTR_RUN_ID:
        odir = os.path.join(odir, config.DISTRIBUTED.RUN_ID)
    makedir(odir)
    return odir


def get_absolute_path(input_path):
    odir = os.path.abspath(input_path)
    makedir(odir)
    return odir


def is_checkpoint_phase(mode_num, mode_frequency, train_phase_idx, num_epochs, mode):
    """
    Determines if a checkpoint should be saved on current epoch. If epoch=1, then
    we check whether to save at current iteration or not.
    """
    if mode == "iteration":
        checkpointing_phase = (mode_num % mode_frequency) == 0
    elif mode == "phase":
        checkpointing_phase = (mode_num % mode_frequency) == 0 or train_phase_idx == (
            num_epochs - 1
        )
    return checkpointing_phase


def has_checkpoint(checkpoint_folder, skip_final: bool = False):
    checkpointed_files = os.listdir(checkpoint_folder)
    checkpoint_exists = False
    for f in checkpointed_files:
        if f.endswith(".torch") and ("model_final" not in f or not skip_final):
            checkpoint_exists = True
            break
    return checkpoint_exists


def has_final_checkpoint(
    checkpoint_folder, final_checkpoint_pattern: str = "model_final"
):
    checkpointed_files = os.listdir(checkpoint_folder)
    torch_files = filter(lambda x: x.endswith(".torch"), checkpointed_files)
    final_files = filter(lambda x: final_checkpoint_pattern in x, torch_files)
    return len(list(final_files)) > 0


def move_checkpoint_to_backend(source, destination, backend):
    success = False
    i = 0
    while (i < 3) and (not success):
        try:
            if backend == "disk":
                ckpt_name = source.split("/")[-1]
                tmp_dir = tempfile.mkdtemp()
                tmp_file = os.path.join(tmp_dir, ckpt_name)
                copy2(source, tmp_file)
                move(tmp_file, destination)
                logging.info(f"Checkpoint saved to {destination}")
                success = True
            else:
                logging.warning(f"{backend} not supported")
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def get_checkpoint_resume_file(checkpoint_folder, config, skip_final: bool = False):
    all_files = os.listdir(checkpoint_folder)
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
            iter_num = int(f.replace(".torch", "").replace(replace_prefix, ""))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filename = f"{replace_prefix}{last_iter}.torch"
        return filename
    else:
        return None


def get_resume_checkpoint(cfg, checkpoint_folder):
    # we check whether there's a checkpoint that already exists
    checkpoint = None
    # if we are overwriting the existing checkpoint, then ski_final=true in has_checkpoint
    # call
    checkpoints_exists = has_checkpoint(
        checkpoint_folder, skip_final=cfg["CHECKPOINT"]["OVERWRITE_EXISTING"]
    )
    if checkpoints_exists and cfg["CHECKPOINT"]["AUTO_RESUME"]:
        checkpoint_device = torch.device("cpu")
        checkpoint_file = get_checkpoint_resume_file(
            checkpoint_folder, cfg, skip_final=cfg["CHECKPOINT"]["OVERWRITE_EXISTING"]
        )
        logging.info(
            f"Resume from file: {os.path.join(checkpoint_folder, checkpoint_file)}"
        )
        checkpoint = load_checkpoint(
            checkpoint_path=os.path.join(checkpoint_folder, checkpoint_file),
            device=checkpoint_device,
        )
    return checkpoint


def _print_state_dict_shapes(state_dict):
    logging.info("Model state_dict:")
    for param_tensor in state_dict.keys():
        logging.info(f"{param_tensor}:\t{state_dict[param_tensor].size()}")


def replace_module_suffix(state_dict, suffix, replace_with=""):
    """
    Replace suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {
        (key.replace(suffix, replace_with, 1) if key.startswith(suffix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def append_module_suffix(state_dict, suffix):
    """
    Append suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {f"{suffix}{key}": val for (key, val) in state_dict.items()}
    return state_dict


def init_model_from_weights(
    model,
    state_dict,
    state_dict_key_name,
    skip_layers=None,
    print_init_layers=True,
    replace_suffix=None,
    append_suffix="trunk.base_model.",
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the finetuning process or when we want to evaluate a model on a range
    of tasks.
    skip_layers:     string : layer names with this key are not copied
    replace_suffix: string : remove these suffixes from the layer names
    print_init_layers:   print whether layer was init or ignored
                    indicates whether the layername was copied or not
    """
    # whether it's a model from somewhere else or a model from this codebase
    if state_dict_key_name and len(state_dict_key_name) > 0:
        assert (
            state_dict_key_name in state_dict.keys()
        ), f"Unknown state dict key: {state_dict_key_name}"
        state_dict = state_dict[state_dict_key_name]
    if state_dict_key_name == "classy_state_dict":
        classy_state_dict = state_dict["base_model"]["model"]
        state_dict = {}
        state_dict.update(classy_state_dict["trunk"])
        state_dict.update(classy_state_dict["heads"])
    if replace_suffix:
        state_dict = replace_module_suffix(state_dict, replace_suffix)
    if append_suffix:
        state_dict = append_module_suffix(state_dict, append_suffix)
    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
            skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (local_rank == 0):
                not_init.append(layername)
                logging.info(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)
            init_layers[layername] = True
            if print_init_layers and (local_rank == 0):
                logging.info(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (local_rank == 0):
                logging.info(f"Not found:\t{layername}")
    ####################### DEBUG ############################
    # _print_state_dict_shapes(model.state_dict())
    return model
