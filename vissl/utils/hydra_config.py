# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import math
import os
import pprint
import re
import sys
from typing import Any, List, NamedTuple, Tuple

import torch
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf, open_dict
from vissl.config import AttrDict, check_cfg_version
from vissl.utils.io import save_file
from vissl.utils.misc import is_augly_available


def save_attrdict_to_disk(cfg: AttrDict):
    from vissl.utils.checkpoint import get_checkpoint_folder

    yaml_output_file = f"{get_checkpoint_folder(cfg)}/train_config.yaml"
    save_file(cfg.to_dict(), yaml_output_file)


class SweepHydraOverrides(NamedTuple):
    overrides: List[Any]
    sweeps: List[List[Any]]

    @classmethod
    def from_overrides(cls, cli_overrides: List[Any]) -> "SweepHydraOverrides":
        """
        Takes an override list and separate the overrides describing
        parameter sweeps from the rest of the overrides.

        Then use those sweeping overrides to generate all possible
        parameter sweep thought grid search.

        Outputs 2 lists:
        - the non sweep overrides
        - all possible sweep combinations
        """

        # Separate the normal overrides from the sweep ones
        overrides = []
        sweep_overrides = []
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=cli_overrides)
        for parsed_override in parsed_overrides:
            if parsed_override.is_sweep_override():
                sweep_overrides.append(parsed_override)
            else:
                overrides.append(parsed_override.input_line)

        # If not parameter sweep specified, return early
        if not sweep_overrides:
            return SweepHydraOverrides(overrides=overrides, sweeps=[])

        # Generate all combinations of sweeps
        sweeps = [[]]
        for parsed_override in sweep_overrides:
            key = parsed_override.key_or_group
            prev_sweeps = sweeps
            sweeps = []
            for combination in prev_sweeps:
                for value in parsed_override.value().list:
                    sweeps.append(combination + [f"{key}={value}"])

        # Return parameters to schedule the hyper-parameter sweep
        return SweepHydraOverrides(overrides=overrides, sweeps=sweeps)


def convert_to_attrdict(
    cfg: DictConfig, cmdline_args: List[Any] = None, dump_config: bool = True
):
    """
    Given the user input Hydra Config, and some command line input options
    to override the config file:
    1. merge and override the command line options in the config
    2. Convert the Hydra OmegaConf to AttrDict structure to make it easy
       to access the keys in the config file
    3. Also check the config version used is compatible and supported in vissl.
       In future, we would want to support upgrading the old config versions if
       we make changes to the VISSL default config structure (deleting, renaming keys)
    4. We infer values of some parameters in the config file using the other
       parameter values.
    """
    if cmdline_args:
        # convert the command line args to DictConfig
        sys.argv = cmdline_args
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # Completion of the "defaults.yaml" for the teacher model used for distillation
    # This avoids repeating the same default options in both:
    # - config.MODEL
    # - cfg.config.DISTILLATION.TEACHER_MODEL
    base_model = copy.deepcopy(cfg.config.MODEL)
    with open_dict(base_model):
        base_model.merge_with(cfg.config.DISTILLATION.TEACHER_MODEL)
        cfg.config.DISTILLATION.TEACHER_MODEL = base_model

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # check the cfg has valid version
    check_cfg_version(cfg)

    # assert the config and infer
    config = cfg.config
    infer_and_assert_hydra_config(config, cfg.engine_name)
    if dump_config:
        save_attrdict_to_disk(config)
    convert_fsdp_dtypes(config)
    return cfg, config


def convert_fsdp_dtypes(config: AttrDict):
    """
    Transform configuration types (primitive types) to VISSL specific types
    """
    # TODO (Quentin) - remove this once FSDP accepts a boolean
    if config["MODEL"]["FSDP_CONFIG"]["compute_dtype"] == "float32":
        config["MODEL"]["FSDP_CONFIG"]["compute_dtype"] = torch.float32
    else:
        config["MODEL"]["FSDP_CONFIG"]["compute_dtype"] = torch.float16


def is_hydra_available():
    """
    Check if Hydra is available. Simply python import to test.
    """
    try:

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def get_hydra_version() -> Tuple[int, ...]:
    import hydra

    return tuple(int(re.findall("\\d+", x)[0]) for x in hydra.__version__.split("."))


def assert_hydra_dependency():
    """
    Check if Hydra is available. Simply python import to test.
    Also verifies whether the version is up to date.
    """
    min_hydra_version = (1, 0, 7)
    min_hydra_version_str = ".".join(str(x) for x in min_hydra_version)
    install_command = f"pip install hydra-core=={min_hydra_version_str}"
    assert is_hydra_available(), f"Make sure to install Hydra: {install_command}"
    upgrade_message = f"Please upgrade Hydra: {install_command}"
    assert get_hydra_version() >= min_hydra_version, upgrade_message


@contextlib.contextmanager
def initialize_hydra_config_module():
    # Backward compatibility with previous hydra versions:
    # In Hydra 1.1 and above, the compose API is not experimental anymore
    if get_hydra_version() >= (1, 1, 0):
        from hydra import initialize_config_module
    else:
        from hydra.experimental import initialize_config_module

    with initialize_config_module(config_module="vissl.config"):
        yield


def hydra_compose(overrides: List[str]):
    # Backward compatibility with previous hydra versions:
    # In Hydra 1.1 and above, the compose API is not experimental anymore
    if get_hydra_version() >= (1, 1, 0):
        from hydra import compose
    else:
        from hydra.experimental import compose
    return compose("defaults", overrides=overrides)


def compose_hydra_configuration(overrides: List[str]):
    """
    Transform the list of overrides provided on the command line
    to an actual VISSL configuration by merging these overrides
    with the defaults configuration of VISSL
    """
    assert_hydra_dependency()

    # Backward compatibility with previous hydra versions:
    # In Hydra 1.1 and above, the compose API is not experimental anymore
    if get_hydra_version() >= (1, 1, 0):
        from hydra import compose, initialize_config_module
    else:
        from hydra.experimental import compose, initialize_config_module

    # Compose the overrides with "vissl/config/defaults.yaml"
    with initialize_config_module(config_module="vissl.config"):
        return compose("defaults", overrides=overrides)


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    if isinstance(cfg, DictConfig):
        if hasattr(cfg, "pretty"):
            # Backward compatibility
            logging.info(cfg.pretty())
        else:
            # Newest version of OmegaConf
            logging.info(OmegaConf.to_yaml(cfg))
    else:
        logging.info(pprint.pformat(cfg))


def resolve_linear_schedule(cfg, param_schedulers):
    """
    For the given composite schedulers, for each linear schedule,
    if the training is 1 node only, the https://arxiv.org/abs/1706.02677 linear
    warmup rule has to be checked if the rule is applicable and necessary.

    We set the end_value = scaled_lr (assuming it's a linear warmup).
    In case only 1 machine is used in training, the start_lr = scaled_lr and then
    the linear warmup is not needed.
    """
    # compute what should be the linear warmup start LR value.
    # this depends on batchsize per node.
    # TODO - why does it not depend on the global batch size?
    num_nodes = cfg.DISTRIBUTED.NUM_NODES
    num_gpus_per_node = cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    bs_per_gpu = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
    batch_size_per_node = bs_per_gpu * num_gpus_per_node
    base_lr = param_schedulers.auto_lr_scaling.base_value
    base_lr_batch_size = param_schedulers.auto_lr_scaling.base_lr_batch_size
    scale_factor = float(batch_size_per_node) / base_lr_batch_size
    start_value = base_lr * scale_factor

    remove_linear_idx = -1
    for idx in range(len(param_schedulers["schedulers"])):
        if param_schedulers["schedulers"][idx]["name"] == "linear":
            param_schedulers["schedulers"][idx]["start_value"] = start_value
            if num_nodes == 1:
                end_value = param_schedulers["schedulers"][idx]["end_value"]
                if start_value <= end_value:
                    # linear schedule is not meaningful as linear warmup is not needed.
                    remove_linear_idx = idx

    # check if linear warmup should be removed as its not meaningul
    if remove_linear_idx >= 0:
        del param_schedulers["schedulers"][remove_linear_idx]
    # if after removing linear warmup, there's only one scheduler, then a composite
    # schedule is no longer needed. The remaining scheduler becomes the primary
    # scheduler
    if len(param_schedulers["schedulers"]) == 1:
        for key, value in param_schedulers["schedulers"][0].items():
            param_schedulers[key] = value
    return param_schedulers


def get_scaled_lr_scheduler(cfg, param_schedulers, scaled_lr):
    """
    Scale learning rate value for different Learning rate types. See infer_learning_rate()
    for how the scaled LR is calculated.

    Values changed for learning rate schedules:
    1. cosine:
        end_value = scaled_lr * (end_value / start_value)
        start_value = scaled_lr and
    2. multistep:
        gamma = values[1] / values[0]
        values = [scaled_lr * pow(gamma, idx) for idx in range(len(values))]
    3. step_with_fixed_gamma
        base_value = scaled_lr
    4. linear:
       end_value = scaled_lr
    5. inverse_sqrt:
       start_value = scaled_lr
    6. constant:
       value = scaled_lr
    7. composite:
        recursively call to scale each composition. If the composition consists of a linear
        schedule, we assume that a linear warmup is applied. If the linear warmup is
        applied, it's possible the warmup is not necessary if the global batch_size is smaller
        than the base_lr_batch_size and in that case, we remove the linear warmup from the
        schedule.
    """
    if "cosine" in param_schedulers["name"]:
        start_value = param_schedulers["start_value"]
        end_value = param_schedulers["end_value"]
        decay_multiplier = end_value / start_value
        param_schedulers["start_value"] = float(scaled_lr)
        param_schedulers["end_value"] = float(scaled_lr * decay_multiplier)
    elif param_schedulers["name"] == "multistep" or param_schedulers["name"] == "step":
        values = param_schedulers["values"]
        gamma = 1.0
        if len(values) > 1:
            gamma = round(values[1] / values[0], 6)
        new_values = []
        for idx in range(len(values)):
            new_values.append(round(float(scaled_lr * pow(gamma, idx)), 8))
        param_schedulers["values"] = new_values
    elif param_schedulers["name"] == "step_with_fixed_gamma":
        param_schedulers["base_value"] = scaled_lr
    elif param_schedulers["name"] == "composite":
        has_linear_warmup = False
        for idx in range(len(param_schedulers["schedulers"])):
            if param_schedulers["schedulers"][idx]["name"] == "linear":
                has_linear_warmup = True
            scheduler = get_scaled_lr_scheduler(
                cfg, param_schedulers["schedulers"][idx], scaled_lr
            )
            param_schedulers["schedulers"][idx] = scheduler
        # in case of composite LR schedule, if there's linear warmup specified,
        # we check if the warmup is meaningful or not. If not, we simplify the
        # schedule.
        if has_linear_warmup:
            resolve_linear_schedule(cfg, param_schedulers)
    elif param_schedulers["name"] == "linear":
        param_schedulers["end_value"] = scaled_lr
    elif param_schedulers["name"] == "inverse_sqrt":
        param_schedulers["start_value"] = scaled_lr
    elif param_schedulers["name"] == "constant":
        param_schedulers["value"] = scaled_lr
    else:
        raise RuntimeError(
            f"Unknow param_scheduler: {param_schedulers['name']}. NOT scaling linearly"
        )
    return param_schedulers


def infer_learning_rate(cfg):
    """
    1) Assert the Learning rate here. LR is scaled as per https://arxiv.org/abs/1706.02677.
    to turn this automatic scaling off,
    set config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=false

    scaled_lr is calculated:
        given base_lr_batch_size = batch size for which the base learning rate is specified,
              base_value = base learning rate value that will be scaled,
              The current batch size is used to determine how to scale the base learning rate
              value.
        scale_factor = (batchsize_per_gpu * world_size) / base_lr_batch_size
        if scaling_type is sqrt, scale factor = sqrt(scale_factor)
        scaled_lr = scale_factor * base_value


    We perform this auto-scaling for head learning rate as well if user wants to use a different
    learning rate for the head

    2) infer the model head params weight decay: if the head should use a different weight
       decay value than the trunk.
       If using different weight decay value for the head, set here. otherwise, the
       same value as trunk will be automatically used.
    """
    if cfg.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale:
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA * world_size
        param_schedulers = cfg.OPTIMIZER.param_schedulers.lr
        base_lr = param_schedulers.auto_lr_scaling.base_value
        base_lr_batch_size = param_schedulers.auto_lr_scaling.base_lr_batch_size
        scaling_type = param_schedulers.auto_lr_scaling.scaling_type
        assert scaling_type in [
            "sqrt",
            "linear",
        ], "Only linear | sqrt scaling_types are supported"

        scale_factor = float(batch_size) / base_lr_batch_size
        if scaling_type == "sqrt":
            scale_factor = math.pow(scale_factor, 0.5)
        scaled_lr = base_lr * scale_factor
        cfg.OPTIMIZER.param_schedulers.lr = get_scaled_lr_scheduler(
            cfg, param_schedulers, scaled_lr
        )

    if not cfg.OPTIMIZER.head_optimizer_params.use_different_lr:
        # if not using the different value for the head, we set the weight decay and LR
        # param scheduler same as the trunk.
        cfg.OPTIMIZER.param_schedulers.lr_head = cfg.OPTIMIZER.param_schedulers.lr
    elif (
        cfg.OPTIMIZER.head_optimizer_params.use_different_lr
        and cfg.OPTIMIZER.param_schedulers.lr_head
        and cfg.OPTIMIZER.param_schedulers.lr_head.auto_lr_scaling.auto_scale
    ):
        # if the user wants a different LR value for the head, then we
        # automatically infer the LR values for the head as well (similar to
        # trunk above)
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA * world_size
        param_schedulers = cfg.OPTIMIZER.param_schedulers.lr_head
        base_lr = param_schedulers.auto_lr_scaling.base_value
        base_lr_batch_size = param_schedulers.auto_lr_scaling.base_lr_batch_size
        scaling_type = param_schedulers.auto_lr_scaling.scaling_type
        assert scaling_type in [
            "sqrt",
            "linear",
        ], "Only linear | sqrt scaling_types are supported"

        scale_factor = float(batch_size) / base_lr_batch_size
        if scaling_type == "sqrt":
            scale_factor = math.pow(scale_factor, 0.5)
        scaled_lr = base_lr * scale_factor
        cfg.OPTIMIZER.param_schedulers.lr_head = get_scaled_lr_scheduler(
            cfg, param_schedulers, scaled_lr
        )

    # for the head, if we want to use a different weight decay value,
    # we verify that the specified weight decay value is valid. Otherwise,
    # we do the inference and set the weight decay value same as the trunk.
    if not cfg.OPTIMIZER.head_optimizer_params.use_different_wd:
        cfg.OPTIMIZER.head_optimizer_params.weight_decay = cfg.OPTIMIZER.weight_decay
    else:
        assert (
            cfg.OPTIMIZER.head_optimizer_params.weight_decay >= 0.0
        ), "weight decay for head should be >=0"
    return cfg


def infer_losses_config(cfg):
    """
    Infer settings for various self-supervised losses. Takes care of setting various loss
    parameters correctly like world size, batch size per gpu, effective global batch size,
    collator etc.
    Each loss has additional set of parameters that can be inferred to ensure smooth
    training in case user forgets to adjust all the parameters.
    """
    train_transforms = cfg.DATA.TRAIN.TRANSFORMS
    total_num_crops = None
    multicrop_crops = []
    for transform in train_transforms:
        if "total_num_crops" in transform:
            total_num_crops = transform["total_num_crops"]
            multicrop_crops = transform["num_crops"]

    # some inference for the Info-NCE loss.
    if "simclr_info_nce_loss" in cfg.LOSS.name:
        cfg.LOSS[cfg.LOSS.name]["buffer_params"]["world_size"] = (
            cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        )

        world_size = cfg.LOSS[cfg.LOSS.name]["buffer_params"]["world_size"]
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        num_positives = 2  # simclr uses 2 copies per image
        cfg.LOSS[cfg.LOSS.name]["buffer_params"]["effective_batch_size"] = (
            num_positives * batch_size * world_size
        )

    # bce_logits_multiple_output_single_target
    if cfg.LOSS.name == "bce_logits_multiple_output_single_target":
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        cfg.LOSS.bce_logits_multiple_output_single_target.world_size = world_size

    # multicrop version of simclr loss
    if cfg.LOSS.name == "multicrop_simclr_info_nce_loss":
        world_size = cfg.LOSS.multicrop_simclr_info_nce_loss.buffer_params.world_size
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        cfg.LOSS.multicrop_simclr_info_nce_loss.buffer_params.world_size = world_size
        cfg.LOSS.multicrop_simclr_info_nce_loss.buffer_params.effective_batch_size = (
            batch_size * world_size
        )
        cfg.LOSS.multicrop_simclr_info_nce_loss.num_crops = (
            total_num_crops or cfg.LOSS.multicrop_simclr_info_nce_loss.num_crops
        )
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the DeepCluster-v2 loss.
    if cfg.LOSS.name == "deepclusterv2_loss":
        cfg.LOSS.deepclusterv2_loss.DROP_LAST = cfg.DATA.TRAIN.DROP_LAST
        cfg.LOSS.deepclusterv2_loss.BATCHSIZE_PER_REPLICA = (
            cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        )
        cfg.LOSS.deepclusterv2_loss.num_crops = (
            total_num_crops or cfg.LOSS.deepclusterv2_loss.num_crops
        )
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the SwAV loss.
    if cfg.LOSS.name == "swav_loss":
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] in {"swav_head", "swav_head_fsdp"}
        assert cfg.DATA.TRAIN.COLLATE_FUNCTION in [
            "multicrop_collator",
            "multicrop_mixup_collator",
            "cutmixup_collator",
        ], (
            "for swav loss, use either a collator from "
            "[multicrop_collator, multicrop_mixup_collator]"
        )
        cfg.LOSS.swav_loss.num_prototypes = cfg.MODEL.HEAD.PARAMS[0][1]["num_clusters"]
        cfg.LOSS.swav_loss.embedding_dim = cfg.MODEL.HEAD.PARAMS[0][1]["dims"][-1]
        cfg.LOSS.swav_loss.num_crops = total_num_crops or cfg.LOSS.swav_loss.num_crops
        from vissl.utils.checkpoint import get_checkpoint_folder

        cfg.LOSS.swav_loss.output_dir = get_checkpoint_folder(cfg)
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        batch_size *= world_size
        queue_length = cfg.LOSS.swav_loss.queue.queue_length
        queue_length -= queue_length % batch_size
        cfg.LOSS.swav_loss.queue.queue_length = queue_length
        cfg.LOSS.swav_loss.queue.local_queue_length = queue_length // world_size

    # some inference for the SwAV momentum loss.
    if cfg.LOSS.name == "swav_momentum_loss":
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] == "swav_head"
        cfg.LOSS.swav_momentum_loss.num_prototypes = cfg.MODEL.HEAD.PARAMS[0][1][
            "num_clusters"
        ]
        cfg.LOSS.swav_momentum_loss.embedding_dim = cfg.MODEL.HEAD.PARAMS[0][1]["dims"][
            -1
        ]

        cfg.LOSS.swav_momentum_loss.num_crops = (
            total_num_crops or cfg.LOSS.swav_momentum_loss.num_crops
        )
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        batch_size *= world_size
        queue_length = cfg.LOSS.swav_momentum_loss.queue.queue_length
        queue_length -= queue_length % batch_size
        cfg.LOSS.swav_momentum_loss.queue.queue_length = queue_length
        cfg.LOSS.swav_momentum_loss.queue.local_queue_length = (
            queue_length // world_size
        )

    # some inference for DINO loss.
    if cfg.LOSS.name == "dino_loss":
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] in {
            "swav_head",
            "dino_head",
            "dino_head_fsdp",
        }
        cfg.LOSS.dino_loss.output_dim = cfg.MODEL.HEAD.PARAMS[0][1]["num_clusters"][0]
        cfg.LOSS.dino_loss.num_crops = total_num_crops or cfg.LOSS.dino_loss.num_crops
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the iBOT loss
    if cfg.LOSS.name == "ibot_loss":
        assert cfg.DATA.TRAIN.COLLATE_FUNCTION == "ibot_multicrop_masking_collator"
        for transform in train_transforms:
            is_vit = "vision_transformer" in cfg.MODEL.TRUNK.NAME
            is_mim_transform = transform["name"] == "MaskedImageModeling"
            if is_mim_transform and is_vit:
                patch_size = cfg.MODEL.TRUNK.VISION_TRANSFORMERS.PATCH_SIZE
                transform["patch_size"] = patch_size

        # TODO(IBOT): the "num_clusters" use only works if the head is
        #  shared between patch and class token (to enhance later)
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] in {"ibot_head"}
        num_clusters = cfg.MODEL.HEAD.PARAMS[0][1]["out_dim"]
        cfg.LOSS.ibot_loss.out_dim = num_clusters
        cfg.LOSS.ibot_loss.patch_out_dim = num_clusters
        cfg.LOSS.ibot_loss.num_epochs = cfg.OPTIMIZER.num_epochs
        cfg.LOSS.ibot_loss.num_global_crops = multicrop_crops[0]
        cfg.LOSS.ibot_loss.num_local_crops = total_num_crops - multicrop_crops[0]

    return cfg


def assert_transforms(cfg):
    for transforms in [cfg.DATA.TRAIN.TRANSFORMS, cfg.DATA.TEST.TRANSFORMS]:
        for transform in transforms:
            if "transform_type" in transform:
                assert transform["transform_type"] in [None, "augly"]

                if transform["transform_type"] == "augly":
                    assert is_augly_available(), "Please pip install augly."


def infer_fsdp_setup(cfg):
    """
    inference for the FSDP settings. Conditions are:
    1) use the FSDP task
    2) use the single param group in the optimizer
    3) if AMP is used, it must be PyTorch AMP
    4) If training SwAV, we automatically set the head to SwAV FSDP head
    4) Inference for the FSDP parameters to ensure the good convergence
    """
    if cfg.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP:
        cfg.TRAINER.TASK_NAME = "self_supervision_fsdp_task"
        cfg.OPTIMIZER.construct_single_param_group_only = True

        # safely set flatten_parameters=True for FSDP trainings.
        cfg["MODEL"]["FSDP_CONFIG"]["flatten_parameters"] = True
        # recommended FSDP settings below for the convergence
        cfg["MODEL"]["FSDP_CONFIG"]["compute_dtype"] = "float32"

        # Inference of optimizer configuration
        if cfg["OPTIMIZER"]["use_larc"]:
            cfg["OPTIMIZER"]["name"] = "sgd_fsdp"
            # if using LARC, we set the flatten_params=False so that we can
            # compute the right params groups
            cfg["MODEL"]["FSDP_CONFIG"]["flatten_parameters"] = False

        # AMP based inference
        if cfg["MODEL"]["AMP_PARAMS"]["USE_AMP"]:
            cfg["MODEL"]["AMP_PARAMS"]["AMP_TYPE"] = "pytorch"
            cfg["MODEL"]["FSDP_CONFIG"]["mixed_precision"] = True
            # setup the compute_dtype and fp32_reduce_scatter
            # based on whether O1 or O2 is desired
            if cfg.MODEL.FSDP_CONFIG["AMP_TYPE"] == "O1":
                cfg["MODEL"]["FSDP_CONFIG"]["compute_dtype"] = "float32"
                cfg["MODEL"]["FSDP_CONFIG"]["fp32_reduce_scatter"] = True
            elif cfg.MODEL.FSDP_CONFIG["AMP_TYPE"] == "O2":
                cfg["MODEL"]["FSDP_CONFIG"]["compute_dtype"] = "float16"
                cfg["MODEL"]["FSDP_CONFIG"]["fp32_reduce_scatter"] = False
        else:
            # if not using AMP, we can't use mixed_precision as it requires PyTorch AMP
            cfg["MODEL"]["FSDP_CONFIG"]["mixed_precision"] = False
            # if mixed_precision=False, FSDP mandates setting fp32_reduce_scatter=False
            cfg["MODEL"]["FSDP_CONFIG"]["fp32_reduce_scatter"] = False

        # Inference of the head in case of training with FSDP
        for i, head_param in enumerate(cfg.MODEL.HEAD.PARAMS):
            if head_param[0] == "swav_head":
                cfg.MODEL.HEAD.PARAMS[i][0] = "swav_head_fsdp"
            if head_param[0] == "eval_mlp":
                cfg.MODEL.HEAD.PARAMS[i][0] = "eval_mlp_fsdp"
            if head_param[0] == "mlp":
                cfg.MODEL.HEAD.PARAMS[i][0] = "mlp_fsdp"

        # Inference of the trunk in case of training with FSDP
        if cfg.MODEL.TRUNK.NAME == "regnet":
            cfg.MODEL.TRUNK.NAME = "regnet_fsdp"

        # Profiling the communication requires some setup for FSDP
        if cfg.PROFILING.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY:
            cfg["MODEL"]["FSDP_CONFIG"]["_TRACK_COMMUNICATIONS"] = True

        logging.info(f"Using the FSDP config: {cfg.MODEL.FSDP_CONFIG}")

    # Delete the AUTO_SETUP_FSDP key since we send the FSDP_CONFIG
    # to FSDP from fairscale which doesn't know about AUTO_SETUP_FSDP
    del cfg.MODEL.FSDP_CONFIG["AUTO_SETUP_FSDP"]
    del cfg.MODEL.FSDP_CONFIG["AMP_TYPE"]

    return cfg


def infer_and_assert_hydra_config(cfg, engine_name: str):
    """
    Infer values of few parameters in the config file using the value of other config parameters
    1. Inferring losses
    2. Auto scale learning rate if user has specified auto scaling to be True.
    3. Infer meter names (model layer name being evaluated) since we support list meters
       that have multiple output and same target. This is very common in self-supervised
       learning where we want to evaluate metric for several layers of the models. VISSL
       supports running evaluation for multiple model layers in a single training run.
    4. Support multi-gpu DDP eval model by attaching a dummy parameter. This is particularly
       helpful for the multi-gpu feature extraction especially when the dataset is large for
       which features are being extracted.
    5. Infer what kind of labels are being used. If user has specified a labels source, we set
       LABEL_TYPE to "standard" (also vissl default), otherwise if no label is specified, we
       set the LABEL_TYPE to "sample_index".
    """
    cfg = infer_losses_config(cfg)
    cfg = infer_learning_rate(cfg)
    assert_transforms(cfg)

    # pass the seed to cfg["MODEL"] so that model init on different nodes can
    # use the same seed.
    # TODO (Min): once FSDP supports sync'ing weights from rank 0, we don't need
    #             this anymore.
    cfg["MODEL"]["_MODEL_INIT_SEED"] = cfg.SEED_VALUE
    cfg["DISTILLATION"]["TEACHER_MODEL"]["_MODEL_INIT_SEED"] = cfg.SEED_VALUE

    # in case of linear evaluation, we often evaluate several layers at a time. For each
    # layer, there's a separate accuracy meter. In such case, we want to output the layer
    # name in the meters output to make it easy to interpret the results. This is
    # currently only supported for cases where we have linear evaluation.
    if cfg.METERS is not None:
        from vissl.models import is_feature_extractor_model

        # Ensure backwards compatibility of cfg.METERS.name.
        meter_name = cfg.METERS.get("name", "")
        if meter_name:
            meter_names = set(cfg.METERS.get("names", []))
            meter_names.add(meter_name)
            cfg.METERS.names = list(meter_names)

        meter_names = cfg.METERS.get("names", [])
        valid_meters = [
            "accuracy_list_meter",
            "mean_ap_list_meter",
            "precision_at_k_list_meter",
            "recall_at_k_list_meter",
        ]

        for meter_name in meter_names:
            if meter_name in valid_meters:
                feat_eval_ops_map = (
                    cfg.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
                )
                all_meter_names = [item[0] for item in feat_eval_ops_map]
                if is_feature_extractor_model(cfg.MODEL):
                    cfg.METERS[meter_name]["num_meters"] = len(feat_eval_ops_map)
                    cfg.METERS[meter_name]["meter_names"] = all_meter_names
                elif engine_name == "extract_label_predictions":
                    if len(feat_eval_ops_map) > 0:
                        cfg.METERS[meter_name]["num_meters"] = len(feat_eval_ops_map)
                        cfg.METERS[meter_name]["meter_names"] = all_meter_names
                    else:
                        # if user is not extracting from multiple layers, we assume
                        # the model head is being used.
                        cfg.METERS[meter_name]["num_meters"] = 1

    # in SSL, during pre-training we don't want to use annotated labels or during feature
    # extraction, we don't have annotated labels for some datasets. In such cases, we set
    # the label type to be just the image index in the dataset, unless the
    # user has specifically provided "zero" as the label type, which is
    # necessary when the CutMixUp collator is being used for self-supervised
    # training.
    if len(cfg.DATA.TRAIN.LABEL_SOURCES) == 0 and cfg.DATA.TRAIN.LABEL_TYPE != "zero":
        cfg.DATA.TRAIN.LABEL_TYPE = "sample_index"
    if len(cfg.DATA.TEST.LABEL_SOURCES) == 0 and cfg.DATA.TEST.LABEL_TYPE != "zero":
        cfg.DATA.TEST.LABEL_TYPE = "sample_index"

    # if the user has specified the model initialization from a params_file, we check if
    # the params_file is a url. If it is, we download the file to a local cache directory
    # and use that instead
    cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE = _download_to_cache_if_url(
        cfg, cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE
    )
    cfg.DISTILLATION.TEACHER_MODEL.WEIGHTS_INIT.PARAMS_FILE = _download_to_cache_if_url(
        cfg, cfg.DISTILLATION.TEACHER_MODEL.WEIGHTS_INIT.PARAMS_FILE
    )

    # ZeRO2: Infer the settings for ShardedDDP which shards the optimizer state
    # and the model weights. For ShardedDDP, we must use the OSS optimizer,
    # set the right task name, use the PyTorch AMP if AMP is used.
    if cfg.MODEL.SHARDED_DDP_SETUP.USE_SDP:
        cfg.OPTIMIZER.use_zero = True
        cfg.TRAINER.TASK_NAME = "self_supervision_sdp_task"
        if cfg.MODEL.AMP_PARAMS.USE_AMP:
            cfg.MODEL.AMP_PARAMS.AMP_TYPE = "pytorch"

    # if we use a zero optimizer, we nest the optimizer related settings under the
    # base_optimizer.
    if cfg.OPTIMIZER.use_zero:
        cfg.OPTIMIZER["base_optimizer"] = cfg.OPTIMIZER.copy()
        cfg.OPTIMIZER.name = "zero"
        del cfg.OPTIMIZER.base_optimizer["param_schedulers"]
        del cfg.OPTIMIZER.base_optimizer["regularize_bn"]
        del cfg.OPTIMIZER.base_optimizer["regularize_bias"]
        del cfg.OPTIMIZER.base_optimizer["num_epochs"]
        del cfg.OPTIMIZER.base_optimizer["use_zero"]
        del cfg.OPTIMIZER.base_optimizer["head_optimizer_params"]

    # Infer fsdp settings
    cfg = infer_fsdp_setup(cfg)

    if cfg.DATA.TRAIN.BASE_DATASET == "generic_ssl":
        assert (
            cfg.DATA.TRAIN.get("TRAIN_PHASES_PER_EPOCH", 1) == 1
        ), "When using the generic_ssl, we must set TRAIN_PHASES_PER_EPOCH = 1."

    if cfg.METERS.model_output_mask:
        assert (
            len(cfg.DATA.TEST.DATA_SOURCES) > 0
        ), "Model output mask is only applicable when there is a test dataset."

        assert (
            cfg.DATA.TEST.BASE_DATASET == "generic_ssl"
        ), "Model output mask is only supported with ssl dataset."

        # Remove CHECK_NAN hooks, as model output masking casts the logits
        # to -inf, which will throw an error from the CHECK_NAN hooks.
        cfg.HOOKS.CHECK_NAN = False

    if cfg.HOOKS.EMA_MODEL.ENABLE_EMA_METERS:
        assert cfg.METERS.get("name", "") or cfg.METERS.get(
            "names", []
        ), "Please specify METER.name or METER.names if you are enabling the EMA_MODEL hook."


def _download_to_cache_if_url(cfg, file_path: str) -> str:
    from vissl.utils.checkpoint import get_checkpoint_folder
    from vissl.utils.io import cache_url, is_url

    if is_url(file_path):
        checkpoint_dir = get_checkpoint_folder(cfg)
        cache_dir = os.path.join(checkpoint_dir, "params_file_cache")
        return cache_url(url=file_path, cache_dir=cache_dir)
    return file_path
