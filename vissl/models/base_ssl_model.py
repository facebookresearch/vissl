# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import logging

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel, register_model
from vissl.models.heads import get_model_head
from vissl.models.trunks import TRUNKS as SUPPORTED_TRUNKS
from vissl.models.trunks.feature_extractor import FeatureExtractorModel
from vissl.utils.checkpoint import (  # noqa
    print_loaded_dict_info,
    print_state_dict_shapes,
)
from vissl.utils.env import get_machine_local_and_dist_rank


@register_model("multi_input_output_model")
class BaseSSLMultiInputOutputModel(ClassyModel):
    def __init__(self, model_config, optimizer_config):
        """
        Class to implement a self-supervised model.
        The model is split into `trunk' that computes features and
        `head' that computes outputs.

        This class supports many usecases
        1. Model producing single output as in standard supervised ImageNet training
        2. Model producing multiple outputs (Multi-task)
        3. Model producing multiple outputs from different features (layers)
           from the trunk
        4. Model that accepts multiple inputs (e.g. image and patches).
        5. Model where the trunk is frozen.

        * How to specify heads?
        For information on heads see the `_set_heads` function

        * What features do `heads' operate on?
            One can specify the `trunk' features that the heads operate on using
            the `EVAL_FEATURES' option.
            e.g., ["res5", "res4"] -> two heads that operate on res5 and res4
                respectively. In this case the trunk needs to map res5 and res4
                to specific features.

        * What inputs do `heads' operate on?
            One can specify the `input' to heads mapping in the list
            INPUTS_TO_HEADS. See the _setup_multi_input() function for details.
        """
        self.config = model_config
        self.optimizer_config = optimizer_config
        super().__init__()
        self.eval_mode = None  # this is just informational
        self.local_rank, _ = get_machine_local_and_dist_rank()
        self.trunk = self._get_trunk()
        self.heads = nn.ModuleList()
        self.head_names = []
        self._set_heads()
        self._setup_multi_input_mapping()

    def multi_input_with_head_mapping_forward(self, batch):
        all_outputs = []
        for input_idx in range(len(self.config.MULTI_INPUT_HEAD_MAPPING)):
            input_key = self.config.MULTI_INPUT_HEAD_MAPPING[input_idx]
            # heads that are used for this input
            heads = self._input_to_head_map[input_key]
            feature_names = self._input_to_eval_features_map[input_key]
            outputs = self._single_input_forward(batch[input_key], feature_names, heads)
            if len(outputs) == 1:
                # single head. do not make nested list
                outputs = outputs[0]
            all_outputs.append(outputs)
        return all_outputs

    def multi_res_input_forward(self, batch):
        assert isinstance(batch, list)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in batch]), return_counts=True
            )[1],
            0,
        )

        feats = []
        start_idx = 0
        for end_idx in idx_crops:
            feat = self.trunk(
                torch.cat(batch[start_idx:end_idx]), self.config.EVAL_FEATURES
            )
            start_idx = end_idx
            assert len(feat) == 1
            feats.append(feat[0])
        feats = [torch.cat(feats)]
        return self.heads_forward(feats, self.heads)

    def _single_input_forward(self, batch, feature_names, heads):
        assert isinstance(batch, torch.Tensor)
        feats = self.trunk(batch, feature_names)
        # if we are interested in extracting the features only.
        if self.config["EXTRACT_FEATURES_ONLY"]:
            return feats
        return self.heads_forward(feats, heads)

    def heads_forward(self, feats, heads):
        # Example case: training linear classifiers on various layers
        if len(feats) == len(heads):
            output = []
            for feat, head in zip(feats, heads):
                output.append(head(feat))
            return output
        # Example case: multiple heads composed in pretext task
        elif (len(heads) > 1) and (len(feats) == 1):
            output = feats[0]
            for head in heads:
                output = head(output)
            # our model is multiple output.
            return [output]
        else:
            logging.error(
                f"Mismatch in #head: {len(heads)} and #features: {len(feats)}"
            )

    def forward(self, batch):

        if len(self.config.MULTI_INPUT_HEAD_MAPPING) > 0:
            # this model accepts multiple types of inputs
            return self.multi_input_with_head_mapping_forward(batch)

        if isinstance(batch, list):
            return self.multi_res_input_forward(batch)

        return self._single_input_forward(batch, self.config.EVAL_FEATURES, self.heads)

    def freeze_head(self):
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = False

    def freeze_trunk(self):
        for param in self.trunk.parameters():
            param.requires_grad = False

    def freeze_head_and_trunk(self):
        # Freeze the full model including the heads and the trunk. In 99% cases,
        # we do not use the pretext head as it is specific to the self-supervised
        # pretext task. But in case of some models like NPID, SimCLR, SwAV, the
        # head is essentially a low dimensional feature projection which we want
        # to use. Hence, we provide utility to freeze the full model.
        self.freeze_trunk()
        self.freeze_head()

    def get_features(self, batch):
        # we don't run the heads and only the trunk. The trunk will already
        # have the feature extractor AvgPool layers and flattened features
        feats = self.trunk(batch)
        return feats

    def _get_trunk(self):
        if self.config.FEATURE_EVAL_MODE:
            self.eval_mode = True
            return FeatureExtractorModel(self.config)
        else:
            self.eval_mode = False
            trunk_name = self.config.TRUNK.NAME
            assert trunk_name in SUPPORTED_TRUNKS, "Trunk unknown"
            return SUPPORTED_TRUNKS[trunk_name](self.config, trunk_name)

    def _set_heads(self):
        """
        This function creates the heads needed by the module.
        HEAD.PARAMS is a list containing parameters for (multiple) heads.
        Each head consist of head_modules that can be composed in different ways.
        * Head Module
            A head_module is specified as a list ["name", kwargs], for example,
            ["mlp", {"dims": [2048, 128]}]

        Examples of Heads one can specify.
        * Simple Head containing single module
          Single Input, Single output
            ["mlp", {"dims": [2048, 128]}]
        * Complex Head containing chain of head modules
          Single Input, Single output
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False, "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}]
            ]
        * Multiple Heads (2 heads)
          Single input, multiple output. Can be used for multi-task learning
            # head 0
            [
                ["mlp", {"dims": [2048, 128]}]
            ],
            # head 1
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False, "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}],
            ]
        * Multiple Heads (5 simple heads)
          Single input, multiple output.
          For example, used in linear evaluation of models
            ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
            ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],
        * Heads can be applied to different types of inputs.
          See `_setup_multi_input_mapping`
        """
        for head_param in self.config.HEAD.PARAMS:
            if isinstance(head_param[0], list):
                # head is composed of several modules
                head_type = []
                head_modules = []
                for idx in range(len(head_param)):
                    head_modules.append(self._make_head_module(head_param[idx]))
                    head_type.append(head_param[idx][0])
                head_name = "->".join(head_type)
                head = nn.Sequential(*head_modules)
            else:
                # head is a single module
                head_name = head_param[0]
                head = self._make_head_module(head_param)
            self.heads.append(head)
            self.head_names.append(head_name)

    def _setup_multi_input_mapping(self):
        """
        Assumptions:
        - This assumes that the same trunk is used to extract features
          for the different types of inputs.
        - One head only operates on one kind of input

        * Specify Input -> Head mapping
           Model accepts multiple types of inputs (in a dictionary)
           MODEL.INPUT_KEYS_TO_HEADS defines a mapping
           between the input_keys to the head that is used for them.
        * Specify Trunk Feature -> Head mapping
           Like in the single input case, the heads can operate on features
           from different layers. In this case, we specify MODEL.EVAL_FEATURES
           to be a list like:
           [
               ["input_key", [ list of features applied to the heads]]
           ]
           For example for a model that applies two heads on images
           and one head on patches
           [
               ["images", ["res5", "res4"]],
               ["patches", ["res3"]],
           ]
        """
        if len(self.config.MULTI_INPUT_HEAD_MAPPING) == 0:
            return

        distinct_input_keys = set(self.config.MULTI_INPUT_HEAD_MAPPING)

        # verify config
        assert len(self.config.EVAL_FEATURES) == len(
            distinct_input_keys
        ), "EVAL_FEATURES must be a nested list with length == input_keys"

        assert len(self.config.MULTI_INPUT_HEAD_MAPPING) == len(
            self.heads
        ), "MULTI_INPUT_HEAD_MAPPING must be a list of length == #heads"

        # create many-to-one mapping from input_key to head
        self._input_to_head_map = {}
        for idx, key in enumerate(self.config.MULTI_INPUT_HEAD_MAPPING):
            if key not in self._input_to_head_map:
                self._input_to_head_map[key] = []
            self._input_to_head_map[key].append(self.heads[idx])

        # create many-to-one mapping from input key to eval features
        self._input_to_eval_features_map = {}
        for input_idx in range(len(self.config.EVAL_FEATURES)):
            key = self.config.EVAL_FEATURES[input_idx][0]
            eval_layer_names = self.config.EVAL_FEATURES[input_idx][1]
            if key in self._input_to_eval_features_map:
                raise ValueError(
                    f"duplicate key {key} \
                    specified for MODEL.EVAL_FEATURES."
                )
            self._input_to_eval_features_map[key] = eval_layer_names

    def _make_head_module(self, head_param):
        head_name = head_param[0]
        head_kwargs = head_param[1]
        head_module = get_model_head(head_name)(self.config, **head_kwargs)
        return head_module

    # we call this on the state.base_model which is not wrapped with DDP.
    # get the model state_dict to checkpoint
    def get_classy_state(self, deep_copy=False):
        trunk_state_dict = self.trunk.state_dict()
        heads_state_dict = self.heads.state_dict()
        model_state_dict = {
            "model": {"trunk": trunk_state_dict, "heads": heads_state_dict}
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        # print_state_dict_shapes(trunk_state_dict)   # DEBUG
        # print_state_dict_shapes_shapes(heads_state_dict)   # DEBUG
        return model_state_dict

    # we call this on the state.base_model which is not wrapped with DDP.
    # load the model from checkpoint
    def set_classy_state(self, state):
        logging.info("Loading Trunk state dict....")
        self.trunk.load_state_dict(state["model"]["trunk"])
        logging.info("Loading Heads state dict....")

        # sometimes, we want to load the partial head only, so strict=False
        self.heads.load_state_dict(state["model"]["heads"], strict=False)
        logging.info("Model state dict loaded!")

        # print debug information about layers loaded
        if self.local_rank == 0:
            # get the model state dict original
            model_state_dict = {}
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
            model_state_dict.update(trunk_state_dict)
            model_state_dict.update(heads_state_dict)

            # get the checkpoint state dict
            checkpoint_state_dict = {}
            checkpoint_state_dict.update(state["model"]["trunk"])
            checkpoint_state_dict.update(state["model"]["heads"])
            params_from_file = self.config["WEIGHTS_INIT"]
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            print_loaded_dict_info(
                model_state_dict, checkpoint_state_dict, skip_layers=skip_layers
            )

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def model_depth(self):
        raise NotImplementedError

    def validate(self, dataset_output_shape):
        raise NotImplementedError
