# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict, List

import torch
from fvcore.common.file_io import PathManager
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.hydra_config import AttrDict
from vissl.utils.io import makedir


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
    assert PathManager.exists(
        config.CHECKPOINT.DIR
    ), "Please specify config.CHECKPOINT.DIR parameter. It should not be None."
    return odir


def is_checkpoint_phase(
    mode_num: int, mode_frequency: int, train_phase_idx: int, num_epochs: int, mode: str
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
        num_epochs (int): total number of epochs in training

    Returns:
        checkpointing_phase (bool): whether the model should be checkpointed or not
    """
    if mode == "iteration":
        checkpointing_phase = (mode_num % mode_frequency) == 0
    elif mode == "phase":
        checkpointing_phase = (mode_num % mode_frequency) == 0 or train_phase_idx == (
            num_epochs - 1
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
    checkpointed_files = PathManager.ls(checkpoint_folder)
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
    checkpointed_files = PathManager.ls(checkpoint_folder)
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
    all_files = PathManager.ls(checkpoint_folder)
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
    Return the checkpoint from which to resume traning. If no checkpoint found,
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
            if (
                not ("heads" in layername)
                or (
                    "heads" in layername
                    and not model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
                )
                or (
                    "heads" in layername
                    and model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
                    and model_config.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD
                )
            ):
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


def check_model_compatibilty(config: AttrDict, state_dict: Dict[str, Any]):
    """
    Given a VISSL model and state_dict, check if the state_dict can be loaded
    to VISSL model (trunk + head) based on the trunk and head prefix that is expected.
    If not compatible, we raise exception.

    Prefix checked for head: `heads.`
    Prefix checked for trunk: `trunk._feature_blocks.` or `trunk.base_model._feature_blocks.`
                              depending on the workflow type (training | evaluation).

    Args:
        config (AttrDict): root config
        state_dict (Dict[str, Any]): state dict that should be checked for compatibility
    """
    from vissl.models import is_feature_extractor_model

    trunk_append_prefix, heads_append_prefix = "trunk._feature_blocks.", "heads."
    if is_feature_extractor_model(config.MODEL):
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
            "Model provided in config.MODEL.WEIGHTS_INIT.PARAMS_FILE is not compatible "
            "with VISSL. Please set config.MODEL.WEIGHTS_INIT.APPEND_PREFIX and "
            "config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX for making model compatible. "
            f"Expected trunk prefix: {trunk_append_prefix}"
        )


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
    if is_feature_extractor_model(config.MODEL):
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
    config: AttrDict,
    model,
    state_dict: Dict[str, Any],
    state_dict_key_name: str,
    skip_layers: List[str],
    replace_prefix=None,
    append_prefix=None,
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the feature evaluation process or when we want to evaluate a model on
    a range of tasks.

    Args:
        config (AttrDict): config file
        model (object): instance of base_ssl_model
        state_dict (Dict): torch.load() of user provided params file path.
        state_dict_key_name (string): key name containing the model state dict
        skip_layers (List(string)): layer names with this key are not copied
        replace_prefix (string): remove these prefixes from the layer names (executed first)
        append_prefix (string): append the prefix to the layer names
                                (executed after replace_prefix)

    Returns:
        model (object): the model initialized from the weights file
    """
    # whether it's a model from somewhere else or a model from this codebase, load the
    # state_dict
    if state_dict_key_name and len(state_dict_key_name) > 0:
        assert (
            state_dict_key_name in state_dict.keys()
        ), f"Unknown state dict key: {state_dict_key_name}"
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
        check_model_compatibilty(config, state_dict)

    # load the checkpoint now
    all_layers = model.state_dict()
    local_rank, _ = get_machine_local_and_dist_rank()
    max_len_model = max(len(key) for key in all_layers.keys())
    for layername in all_layers.keys():
        if len(skip_layers) > 0 and any(item in layername for item in skip_layers):
            if local_rank == 0:
                logging.info(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            # if we are initializing the heads and the feature eval mode is on, we check
            # if we are evaluating the heads as well or not. If not, we don't initialize
            # the heads. Otherwise we initialize the heads.
            if (
                not ("heads" in layername)
                or (
                    "heads" in layername
                    and not config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
                )
                or (
                    "heads" in layername
                    and config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
                    and config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD
                )
            ):
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
            else:
                if local_rank == 0:
                    logging.info(f"Ignored layer:\t{layername}")
        else:
            if local_rank == 0:
                logging.info(f"Not found:\t\t{layername}, not initialized")
    if local_rank == 0:
        extra_layers = []
        # go through the checkpoint state_dict and print what extra layers exist in checkpoint
        for layername in state_dict.keys():
            if layername not in all_layers:
                extra_layers.append(layername)
        logging.info(f"Extra layers not loaded from checkpoint: {extra_layers}")

    ####################### DEBUG ############################
    # print_state_dict_shapes(model.state_dict())
    return model
