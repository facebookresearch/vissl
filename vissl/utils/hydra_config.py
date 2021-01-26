# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint
import sys
from typing import Any, List

from omegaconf import DictConfig, OmegaConf
from vissl.config import check_cfg_version


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed like attributes (as well as normally).
    Credits: https://aiida.readthedocs.io/projects/aiida-core/en/latest/_modules/aiida/common/extendeddicts.html#AttributeDict  # noqa
    """

    def __init__(self, dictionary):
        """
        Recursively turn the `dict` and all its nested dictionaries into `AttrDict` instance.
        """
        super().__init__()

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            else:
                self[key] = value

    def __getattr__(self, key):
        """
        Read a key as an attribute.

        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            return self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __setattr__(self, key, value):
        """
        Set a key as an attribute.
        """
        self[key] = value

    def __delattr__(self, key):
        """
        Delete a key as an attribute.

        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            del self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __getstate__(self):
        """
        Needed for pickling this class.
        """
        return self.__dict__.copy()

    def __setstate__(self, dictionary):
        """
        Needed for pickling this class.
        """
        self.__dict__.update(dictionary)

    def __deepcopy__(self, memo=None):
        """
        Deep copy.
        """
        from copy import deepcopy

        if memo is None:
            memo = {}
        retval = deepcopy(dict(self))
        return self.__class__(retval)

    def __dir__(self):
        return self.keys()


def convert_to_attrdict(cfg: DictConfig, cmdline_args: List[Any] = None):
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

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # check the cfg has valid version
    check_cfg_version(cfg)

    # assert the config and infer
    config = cfg.config
    assert_hydra_conf(config)
    return cfg, config


def is_hydra_available():
    """
    Check if Hydra is available. Simply python import to test.
    """
    try:
        import hydra  # NOQA

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    if isinstance(cfg, DictConfig):
        logging.info(cfg.pretty())
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
    Scale learning rate value for different Learning rate types. See assert_learning_rate()
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


def assert_learning_rate(cfg):
    """
    1) Assert the Learning rate here. LR is scaled as per https://arxiv.org/abs/1706.02677.
    to turn this automatic scaling off,
    set config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=false

    scaled_lr is calculated:
        given base_lr_batch_size = batch size for which the base learning rate is specified,
              base_value = base learning rate value that will be scaled,
              The current batch size is used to determine how to scale the base learning rate
              value.
        scaled_lr = ((batchsize_per_gpu * world_size) * base_value ) / base_lr_batch_size

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
        scale_factor = float(batch_size) / base_lr_batch_size
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
        # if the user wants a different LR value for the head, then we automatically
        # infer the LR values for the head as well (similar to trunk above)
        world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
        batch_size = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA * world_size
        param_schedulers = cfg.OPTIMIZER.param_schedulers.lr_head
        base_lr = param_schedulers.auto_lr_scaling.base_value
        base_lr_batch_size = param_schedulers.auto_lr_scaling.base_lr_batch_size
        scale_factor = float(batch_size) / base_lr_batch_size
        scaled_lr = base_lr * scale_factor
        cfg.OPTIMIZER.param_schedulers.lr_head = get_scaled_lr_scheduler(
            cfg, param_schedulers, scaled_lr
        )

    # for the head, if we want to use a different weight decay value, we verify that
    # the specified weight decay value is valid. Otherwise, we do the inference
    # and set the weight decay value same as the trunk.
    if not cfg.OPTIMIZER.head_optimizer_params.use_different_wd:
        cfg.OPTIMIZER.head_optimizer_params.weight_decay = cfg.OPTIMIZER.weight_decay
    else:
        assert (
            cfg.OPTIMIZER.head_optimizer_params.weight_decay >= 0.0
        ), "weight decay for head should be >=0"
    return cfg


def assert_losses(cfg):
    """
    Infer settings for various self-supervised losses. Takes care of setting various loss
    parameters correctly like world size, batch size per gpu, effective global batch size,
    collator etc.
    Each loss has additional set of parameters that can be inferred to ensure smooth
    training in case user forgets to adjust all the parameters.
    """
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
        total_num_crops = cfg.DATA.TRAIN.TRANSFORMS[0]["total_num_crops"]
        cfg.LOSS.multicrop_simclr_info_nce_loss.buffer_params.world_size = world_size
        cfg.LOSS.multicrop_simclr_info_nce_loss.buffer_params.effective_batch_size = (
            batch_size * world_size
        )
        cfg.LOSS.multicrop_simclr_info_nce_loss.num_crops = total_num_crops
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the DeepCluster-v2 loss.
    if cfg.LOSS.name == "deepclusterv2_loss":
        cfg.LOSS.deepclusterv2_loss.DROP_LAST = cfg.DATA.TRAIN.DROP_LAST
        cfg.LOSS.deepclusterv2_loss.BATCHSIZE_PER_REPLICA = (
            cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
        )
        cfg.LOSS.deepclusterv2_loss.num_crops = cfg.DATA.TRAIN.TRANSFORMS[0][
            "total_num_crops"
        ]
        cfg.DATA.TRAIN.COLLATE_FUNCTION = "multicrop_collator"

    # some inference for the SwAV loss.
    if cfg.LOSS.name == "swav_loss":
        assert len(cfg.MODEL.HEAD.PARAMS) == 1
        assert cfg.MODEL.HEAD.PARAMS[0][0] == "swav_head"
        assert cfg.DATA.TRAIN.COLLATE_FUNCTION in [
            "multicrop_collator",
            "multicrop_mixup_collator",
        ], (
            "for swav loss, use either a collator from "
            "[multicrop_collator, multicrop_mixup_collator]"
        )
        cfg.LOSS.swav_loss.num_prototypes = cfg.MODEL.HEAD.PARAMS[0][1]["num_clusters"]
        cfg.LOSS.swav_loss.embedding_dim = cfg.MODEL.HEAD.PARAMS[0][1]["dims"][-1]
        cfg.LOSS.swav_loss.num_crops = cfg.DATA.TRAIN.TRANSFORMS[0]["total_num_crops"]
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
        cfg.LOSS.swav_momentum_loss.num_crops = cfg.DATA.TRAIN.TRANSFORMS[0][
            "total_num_crops"
        ]
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
    return cfg


def assert_hydra_conf(cfg):
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
    cfg = assert_losses(cfg)
    cfg = assert_learning_rate(cfg)

    # in case of linear evaluation, we often evaluate several layers at a time. For each
    # layer, there's a separate accuracy meter. In such case, we want to output the layer
    # name in the meters output to make it easy to interpret the results. This is
    # currently only supported for cases where we have linear evaluation.
    if cfg.METERS is not None:
        from vissl.models import is_feature_extractor_model

        meter_name = cfg.METERS.get("name", "")
        valid_meters = ["accuracy_list_meter", "mean_ap_list_meter"]
        if meter_name:
            if meter_name in valid_meters and is_feature_extractor_model(cfg.MODEL):
                cfg.METERS[meter_name]["num_meters"] = len(
                    cfg.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
                )
                cfg.METERS[meter_name]["meter_names"] = [
                    item[0]
                    for item in cfg.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
                ]

    # in case of feature evaluation mode, we freeze the trunk. The Feature evaluation mode
    # is used for the feature extraction of trunk as well. VISSL supports distributed feature
    # extraction to speed up the extraction time. Since the model needs to be DDP for the
    # distributed extraction, we need some dummy parameters in the model otherwise model
    # can't be converted to DDP. So we attach some dummy head to the model.
    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    if (
        cfg.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
        and cfg.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY
        and cfg.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY
        and world_size > 1
        and len(cfg.MODEL.HEAD.PARAMS) == 0
    ):
        cfg.MODEL.HEAD.PARAMS = [["mlp", {"dims": [2048, 1000]}]]

    # in SSL, during pre-training we don't want to use annotated labels or during feature
    # extraction, we don't have annotated labels for some datasets. In such cases, we set
    # the label type to be just the image index in the dataset.
    if len(cfg.DATA.TRAIN.LABEL_SOURCES) == 0:
        cfg.DATA.TRAIN.LABEL_TYPE = "sample_index"
    if len(cfg.DATA.TEST.LABEL_SOURCES) == 0:
        cfg.DATA.TEST.LABEL_TYPE = "sample_index"

    # if the user has specified the model initialization from a params_file, we check if
    # the params_file is a url. If it is, we download the file to a local cache directory
    # and use that instead
    from vissl.utils.checkpoint import get_checkpoint_folder
    from vissl.utils.io import cache_url, is_url

    if is_url(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE):
        checkpoint_dir = get_checkpoint_folder(cfg)
        cache_dir = f"{checkpoint_dir}/params_file_cache/"
        cached_url_path = cache_url(
            url=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE, cache_dir=cache_dir
        )
        cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE = cached_url_path
