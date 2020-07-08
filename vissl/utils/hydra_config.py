# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint
import sys
from typing import Any, List

from omegaconf import DictConfig, OmegaConf


class AttrDict(dict):
    """
    Dictionary class which also support attribute access.
    Credits: https://stackoverflow.com/a/38034502
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_dict(data):
        """
        Construct nested AttrDicts from nested dictionaries.
        """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_dict(data[key]) for key in data})


def convert_to_attrdict(cfg: DictConfig, cmdline_args: List[Any] = None):
    if cmdline_args:
        # convert the command line args to DictConfig
        sys.argv = cmdline_args
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_dict(cfg)
    config = cfg.config

    # assert the config and infer
    assert_hydra_conf(config)
    return cfg, config


def is_hydra_available():
    try:
        import hydra  # NOQA

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def print_cfg(cfg):
    logging.info("Training with config:")
    if isinstance(cfg, DictConfig):
        logging.info(cfg.pretty())
    else:
        logging.info(pprint.pformat(cfg))


def resolve_linear_schedule(cfg, param_schedulers):
    # compute what should be the linear warmup start LR value.
    # this depends on batchsize per node.
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
    if param_schedulers["name"] == "cosine":
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
    elif param_schedulers["name"] == "constant":
        param_schedulers["value"] = scaled_lr
    else:
        raise RuntimeError(
            f"Unknow param_scheduler: {param_schedulers['name']}. NOT scaling linearly"
        )
    return param_schedulers


def assert_hydra_conf(cfg):
    # some inference for the Info NCE loss.
    if "simclr_info_nce_loss" in cfg.CRITERION.name:
        cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.BUFFER_PARAMS.WORLD_SIZE = (
            cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        )

        world_size = cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.BUFFER_PARAMS.WORLD_SIZE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        num_positives = 2  # simclr uses 2 copies per image
        cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.BUFFER_PARAMS.EFFECTIVE_BATCH_SIZE = (
            num_positives * batch_size * world_size
        )

    # multicrop version of simclr loss
    if "multicrop_simclr_info_nce_loss" in cfg.CRITERION.name:
        world_size = cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.BUFFER_PARAMS.WORLD_SIZE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        total_nmb_crops = cfg.DATA.TRAIN.TRANSFORMS[0]["total_nmb_crops"]
        cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.BUFFER_PARAMS.EFFECTIVE_BATCH_SIZE = (
            batch_size * world_size
        )
        cfg.CRITERION.SIMCLR_INFO_NCE_LOSS.MULTI_CROP_PARAMS.NMB_CROPS = total_nmb_crops
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the DeepCluster-v2 loss.
    if cfg.CRITERION.name == "deepclusterv2_loss":
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.DROP_LAST = cfg.DATA.TRAIN.DROP_LAST
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.BATCHSIZE_PER_REPLICA = (
            cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        )
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.NMB_CROPS = cfg.DATA.TRAIN.TRANSFORMS[0][
            "total_nmb_crops"
        ]
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the SwAV loss.
    if cfg.CRITERION.name == "swav_loss":
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] == "swav_head"
        cfg.CRITERION.SWAV_LOSS.NMB_PROTOTYPES = cfg.MODEL.HEAD.PARAMS[0][1][
            "nmb_clusters"
        ]
        cfg.CRITERION.SWAV_LOSS.EMBEDDING_DIM = cfg.MODEL.HEAD.PARAMS[0][1]["dims"][-1]
        cfg.CRITERION.SWAV_LOSS.NMB_CROPS = cfg.DATA.TRAIN.TRANSFORMS[0][
            "total_nmb_crops"
        ]
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        batch_size *= world_size
        queue_length = cfg.CRITERION.SWAV_LOSS.QUEUE.QUEUE_LENGTH
        queue_length -= queue_length % batch_size
        cfg.CRITERION.SWAV_LOSS.QUEUE.QUEUE_LENGTH = queue_length
        cfg.CRITERION.SWAV_LOSS.QUEUE.LOCAL_QUEUE_LENGTH = queue_length // world_size

    # assert the Learning rate here. LR is scaled as per https://arxiv.org/abs/1706.02677.
    # to turn this automatic scaling off,
    # set config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=false
    if cfg.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale:
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA * world_size
        param_schedulers = cfg.OPTIMIZER.param_schedulers.lr
        base_lr = param_schedulers.auto_lr_scaling.base_value
        base_lr_batch_size = param_schedulers.auto_lr_scaling.base_lr_batch_size
        scale_factor = float(batch_size) / base_lr_batch_size
        scaled_lr = base_lr * scale_factor
        cfg.OPTIMIZER.param_schedulers.lr = get_scaled_lr_scheduler(
            cfg, param_schedulers, scaled_lr
        )
