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
        raise Exception(
            "Please specify config.CHECKPOINT.DIR parameter. It should not be None."
        )
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


def print_state_dict_shapes(state_dict):
    logging.info("Model state_dict:")
    for param_tensor in state_dict.keys():
        logging.info(f"{param_tensor}:\t{state_dict[param_tensor].size()}")


def print_loaded_dict_info(model_state_dict, state_dict, skip_layers=None):
    """
    Print what layers were loaded, what layers were ignored/skipped/not found
    when initializing a model from a specified model params file.
    """
    extra_layers = []
    max_len_model = max(len(key) for key in model_state_dict.keys())
    # go through the model layers and print what layer is loaded/not loaded/skipped
    for layername in model_state_dict.keys():
        if skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0:
            logging.info(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            logging.info(
                f"Loaded: {layername: <{max_len_model}} of "
                f"shape: {model_state_dict[layername].size()} from checkpoint"
            )
        else:
            logging.info(f"Not found:\t\t{layername}, not initialized")

    # go through the checkpoint state_dict and print what extra layers exist in checkpoint
    for layername in state_dict.keys():
        if layername not in model_state_dict:
            extra_layers.append(layername)
    logging.info(f"Extra layers not loaded from checkpoint: {extra_layers}")


def replace_module_prefix(state_dict, prefix, replace_with=""):
    """
    Replace prefixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {
        (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def append_module_prefix(state_dict, prefix):
    """
    Append prefixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {f"{prefix}{key}": val for (key, val) in state_dict.items()}
    return state_dict


def check_model_compatibilty(config, state_dict):
    trunk_append_prefix, heads_append_prefix = "trunk._feature_blocks.", "heads."
    if config.MODEL.FEATURE_EVAL_MODE:
        trunk_append_prefix = "trunk.base_model._feature_blocks."

    is_compatible = True
    for layername in state_dict.keys():
        if not (
            layername.startswith(trunk_append_prefix)
            or layername.startswith(heads_append_prefix)
        ):
            is_compatible = False
            break
    if not is_compatible:
        raise Exception(
            "Model provided in config.MODEL.PARAMS_FILE.PATH is not compatible with VISSL. "
            "Please set config.MODEL.PARAMS_FILE.APPEND_PREFIX and "
            "config.MODEL.PARAMS_FILE.REMOVE_PREFIX for making model compatible. "
            f"Expected trunk prefix: {trunk_append_prefix}"
        )


def get_checkpoint_model_state_dict(config, state_dict):
    """
    Given a specified pre-trained VISSL model (composed of head and trunk),
    we get the state_dict that can be loaded by appending prefixes to model and trunk.
    """
    classy_state_dict = state_dict["base_model"]["model"]
    trunk_append_prefix, heads_append_prefix = "trunk.", "heads."
    if config.MODEL.FEATURE_EVAL_MODE:
        trunk_append_prefix = "trunk.base_model."

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


def init_model_from_weights(
    config,
    model,
    state_dict,
    state_dict_key_name,
    skip_layers=None,
    replace_prefix=None,
    append_prefix=None,
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the feature evaluation process or when we want to evaluate a model on
    a range of tasks.
    config:                AttrDict: config file
    model:                 object: instance of base_ssl_model
    state_dict:            Dict: torch.load() of user provided params file path.
    state_dict_key_name:   string: key name containing the model state dict
    skip_layers:           string : layer names with this key are not copied
    replace_prefix:        string : remove these prefixes from the layer names (executed first)
    append_prefix:         string : append the prefix to the layer names (executed after replace_prefix)
    """
    # whether it's a model from somewhere else or a model from this codebase, load the
    # state_dict
    if state_dict_key_name and len(state_dict_key_name) > 0:
        assert (
            state_dict_key_name in state_dict.keys()
        ), f"Unknown state dict key: {state_dict_key_name}"
        state_dict = state_dict[state_dict_key_name]

    if state_dict_key_name == "classy_state_dict":
        state_dict = get_checkpoint_model_state_dict(config, state_dict)
    else:
        # make any corrections to the layer names to load checkpoint successfully
        if replace_prefix:
            state_dict = replace_module_prefix(state_dict, replace_prefix)
        if append_prefix:
            state_dict = append_module_prefix(state_dict, append_prefix)
        check_model_compatibilty(config, state_dict)

    # load the checkpoint now
    all_layers = model.state_dict()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    for layername in all_layers.keys():
        if skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0:
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)
    if local_rank == 0:
        print_loaded_dict_info(
            model_state_dict=all_layers, state_dict=state_dict, skip_layers=skip_layers
        )

    ####################### DEBUG ############################
    # print_state_dict_shapes(model.state_dict())
    return model
