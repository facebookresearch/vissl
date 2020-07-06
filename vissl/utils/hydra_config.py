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
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multires_collator"

    # some inference for the DeepCluster-v2 loss.
    if cfg.CRITERION.name == "deepclusterv2_loss":
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.DROP_LAST = cfg.DATA.TRAIN.DROP_LAST
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.BATCHSIZE_PER_REPLICA = (
            cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        )
        cfg.CRITERION.DEEPCLUSTERV2_LOSS.NMB_CROPS = cfg.DATA.TRAIN.TRANSFORMS[0][
            "total_nmb_crops"
        ]
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multires_collator"

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
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multires_collator"
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        batch_size *= world_size
        queue_length = cfg.CRITERION.SWAV_LOSS.QUEUE.QUEUE_LENGTH
        queue_length -= queue_length % batch_size
        cfg.CRITERION.SWAV_LOSS.QUEUE.QUEUE_LENGTH = queue_length
        cfg.CRITERION.SWAV_LOSS.QUEUE.LOCAL_QUEUE_LENGTH = queue_length // world_size
