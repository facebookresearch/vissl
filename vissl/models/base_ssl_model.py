# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, Iterable, List, Union

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel, ClassyModelHeadExecutorWrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models import register_model
from vissl.models.heads import get_model_head, SwAVPrototypesHead
from vissl.models.model_helpers import (
    get_trunk_output_feature_names,
    is_feature_extractor_model,
)
from vissl.models.multiple_input_mapping import MultipleInputMapping
from vissl.models.trunks import get_model_trunk
from vissl.models.trunks.feature_extractor import FeatureExtractorModel
from vissl.utils.checkpoint import (
    CheckpointLoader,
    init_model_from_consolidated_weights,
    should_init_head_weights,
)
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.fsdp_utils import fsdp_recursive_reset_lazy_init
from vissl.utils.misc import set_torch_seed


class VisslClassyModelHeadExecutorWrapper(ClassyModelHeadExecutorWrapper):
    """
    Temporary fix for ClassyVision:
    - ClassyModel overrides via meta class the __setattr__ and __getattr__
      but the __delattr__ is not overridden
    - This leads to a crash when wrapping a model with FSDP because FSDP
      uses extensively setattr, getattr, hasattr and delattr
    - This also leads to asymmetries in which hasattr will return True but
      delattr will crash
    """

    def __delattr__(self, name):
        if name not in ["classy_model", "forward"] and hasattr(self, "classy_model"):
            delattr(self.classy_model, name)
        else:
            super().__delattr__(name)


@register_model("multi_input_output_model")
class BaseSSLMultiInputOutputModel(ClassyModel):
    """
    Class to implement a Self-Supervised model.
    The model is split into `trunk' that computes features and `head' that
    computes outputs (projections, classifications etc)

    This class supports many use cases:
    1. Model producing single output as in standard supervised ImageNet training
    2. Model producing multiple outputs (Multi-task)
    3. Model producing multiple outputs from different features (layers)
       from the trunk (useful in linear evaluation of features from several model layers)
    4. Model that accepts multiple inputs (e.g. image and patches as in PIRL appraoch).
    5. Model where the trunk is frozen.
    6. Model that supports multiple resolutions inputs as in SwAV

    * How to specify heads?
        For information on heads see the `_get_heads()` function

    * What inputs do `heads' operate on?
        One can specify the `input' to heads mapping in the list
        MULTI_INPUT_HEAD_MAPPING. See the _setup_multi_input_head_mapping()
        function for details.
    """

    # Temporary fix around ClassyVision to support delattr property
    wrapper_cls = VisslClassyModelHeadExecutorWrapper

    def __init__(self, model_config, optimizer_config):
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        super().__init__()
        self.eval_mode = None  # this is just informational
        self.local_rank, _ = get_machine_local_and_dist_rank()
        self.trunk = self._get_trunk()
        self.heads = nn.ModuleList()
        self.head_names = []
        self._output_feature_names = get_trunk_output_feature_names(self.model_config)
        self._get_heads()
        self.multi_input_mapping = MultipleInputMapping.from_config(self.model_config)

    def multi_input_with_head_mapping_forward(self, batch):
        """
        Perform forward pass (trunk + heads) separately on each input and return the model
        output on all inputs as a list.
        """
        all_outputs = []
        for input_key in self.multi_input_mapping.input_keys:
            if input_key in batch:
                head = self.heads[self.multi_input_mapping.head_index[input_key]]
                feat_names = self.multi_input_mapping.feat_names[input_key]
                additional_data = {
                    key: batch[key]
                    for key in self.multi_input_mapping.additional_keys[input_key]
                    if key in batch
                }
                outputs = self.single_input_forward(
                    batch[input_key], feat_names, [head], **additional_data
                )
                if len(outputs) == 1:
                    # single head. do not make nested list
                    outputs = outputs[0]
                all_outputs.append(outputs)
        return all_outputs

    def multi_res_input_forward(self, batch, feature_names):
        """
        Perform forward pass separately on each resolution input.
        The inputs corresponding to a single resolution are clubbed and single
        forward is run on the same resolution inputs. Hence we do several
        forward passes = number of different resolutions used. We then
        concatenate all the output features. Then run the head forward on the
        concatenated features.
        """
        assert isinstance(batch, list)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in batch]), return_counts=True
            )[1],
            0,
        )

        feats = []
        start_idx = 0
        # in order to optimize memory usage, we can do single pass per
        # crop as well. Set the flag to be true.
        if self.model_config.SINGLE_PASS_EVERY_CROP:
            idx_crops = torch.Tensor(list(range(1, 1 + len(batch)))).int()
        for end_idx in idx_crops:
            # TODO - factorize with "single_input_forward"
            feat = self.trunk(torch.cat(batch[start_idx:end_idx]), feature_names)
            start_idx = end_idx
            assert len(feat) == 1
            feats.append(feat[0])
        feats = [torch.cat(feats)]
        return self.heads_forward(feats, self.heads)

    def single_input_forward(
        self,
        batch: Union[torch.Tensor, MultiDimensionalTensor],
        feature_names: List[str],
        heads: Union[nn.ModuleList, Iterable[nn.Module]],
        **kwargs,
    ):
        """
        Simply run the trunk and heads forward on the input tensor. We run the trunk
        first and then the heads on the trunk output.
        If the model is trunk feature extraction only, then we simply return the output
        of the trunk.
        """
        assert isinstance(batch, (torch.Tensor, MultiDimensionalTensor)), type(batch)
        feats = self.trunk(batch, feature_names, **kwargs)
        # if we are interested in evaluating the trunk only, we return the output of the trunk
        # and don't forward through the heads
        if (
            self.model_config["FEATURE_EVAL_SETTINGS"]["EVAL_MODE_ON"]
            and self.model_config["FEATURE_EVAL_SETTINGS"][
                "EXTRACT_TRUNK_FEATURES_ONLY"
            ]
        ):
            return feats
        return self.heads_forward(feats, heads)

    def heads_forward(self, feats, heads):
        """
        Run the forward of the head on the trunk output features.
        We have 2 cases:
            1. #heads = #feats -> example training linear classifiers on various layers.
               We run one head on the corresponding feature.
            2. #feats = 1 and #heads > 1 -> head consists of many layers to be run sequentially.
               #outputs = 1
        """
        # Example case: training linear classifiers on various layers
        if len(feats) == len(heads):
            output = []
            for feat, head in zip(feats, heads):
                output.append(head(feat))
            return output
        # Example case: Head consisting of several layers
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
        """
        Main forward of the model. Depending on the model type the calls are patched
        to the suitable function.
        """

        # Used for PIRL / iBOT:
        # this model accepts multiple types of inputs and a separate
        # trunk config or head is applied to each model output of a given input
        if self.multi_input_mapping is not None:
            return self.multi_input_with_head_mapping_forward(batch)

        if isinstance(batch, list):
            return self.multi_res_input_forward(batch, self._output_feature_names)

        return self.single_input_forward(batch, self._output_feature_names, self.heads)

    def freeze_head(self):
        """
        Freeze the model head by setting requires_grad=False for all the parameters
        """
        logging.info("Freezing model heads...")
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = False

    def freeze_trunk(self):
        """
        Freeze the model trunk by setting requires_grad=False for all the parameters
        """
        logging.info("Freezing model trunk...")
        for param in self.trunk.parameters():
            param.requires_grad = False

    def freeze_head_and_trunk(self):
        """
        Freeze the full model including the heads and the trunk. In 99% cases,
        we do not use the pretext head as it is specific to the self-supervised
        pretext task. But in case of some models like NPID, SimCLR, SwAV, the
        head is essentially a low dimensional feature projection which we want
        to use. Hence, we provide utility to freeze the full model.
        """
        logging.info("Freezing model...")
        self.freeze_trunk()
        self.freeze_head()

    def is_fully_frozen_model(self):
        """
        Look at all the parameters of the model (trunk + heads) and check if there
        is any trainable parameter. if not, the model is completely frozen.
        """
        trunk_params_list = self.trunk.parameters()
        heads_params_list = self.heads.parameters()
        trunk_trainable_params = list(
            filter(lambda x: x.requires_grad, trunk_params_list)
        )
        heads_trainable_params = list(
            filter(lambda x: x.requires_grad, heads_params_list)
        )
        if len(trunk_trainable_params) == 0 and len(heads_trainable_params) == 0:
            return True
        return False

    def get_features(self, batch):
        """
        Run the trunk forward on the input batch. This give us the features
        from the trunk at several layers of the model.

        In case of feature extraction, we don't run the heads and only the trunk.
        The trunk will already have the feature extractor Pooling layers and flattened
        features attached. feature extractor heads are part of the trunk already.
        """
        feats = self.trunk(batch)
        return feats

    def _get_trunk(self):
        """
        Construct the model trunk given the architecture specified

        The trunks could be convnet (AlexNet, ResNe(X)t, RegNet,...etc), transformers etc.
        """
        # if we are going to evaluate trunk only we shift to feature extractor backbone
        if is_feature_extractor_model(self.model_config):
            self.eval_mode = True
            return FeatureExtractorModel(self.model_config)
        else:
            self.eval_mode = False
            trunk_name = self.model_config.TRUNK.NAME
            return get_model_trunk(trunk_name)(self.model_config, trunk_name)

    def _build_head_module(self, head_param):
        """
        Given the head, contruct the head.

        Args:
            head_param: The head param is a list containing:
                        head_param = [
                            head_name : str,
                            head_settings: dict containing head settings
                        ]

                        Example:
                            head_param = [
                                "mlp",
                                {"dims": [2048, 128]}
                            ]
        Returns:
            pytorch module for the head
        """
        head_name, head_kwargs = head_param[0], head_param[1]
        head_module = get_model_head(head_name)(self.model_config, **head_kwargs)
        return head_module

    def _get_heads(self):
        """
        This function creates the heads needed by the module.
        HEAD.PARAMS is a list containing parameters for (multiple) heads.
        Each head consist of head_modules that can be composed in different ways.

        * Head Module
            A head_module is specified as a list ["name", kwargs], for example,
            ["mlp", {"dims": [2048, 128]}]

        * Heads can be applied to different types of inputs.
          See `_setup_multi_input_head_mapping`

        Examples of Heads one can specify:
        * Case1: Simple Head containing single module - Single Input, Single output
            ["mlp", {"dims": [2048, 128]}]

        * Case2: Complex Head containing chain of head modules
          Single Input, Single output
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False, "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}]
            ]

        * Case3: Multiple Heads (example 2 heads) - Single input, multiple output
            Can be used for multi-task learning
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

        * Case4: Multiple Heads (example 5 simple heads) - Single input, multiple output.
          For example, used in linear evaluation of models
            ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
            ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],

        """
        for head_param in self.model_config.HEAD.PARAMS:
            if isinstance(head_param[0], list):
                # head is composed of several modules
                head_type, head_modules = [], []
                for idx in range(len(head_param)):
                    with set_torch_seed(self.model_config._MODEL_INIT_SEED + idx):
                        head_modules.append(self._build_head_module(head_param[idx]))
                    head_type.append(head_param[idx][0])
                head_name = "->".join(head_type)
                head = nn.Sequential(*head_modules)
            else:
                # head is a single module
                head_name = head_param[0]
                with set_torch_seed(self.model_config._MODEL_INIT_SEED):
                    head = self._build_head_module(head_param)
            self.heads.append(head)
            self.head_names.append(head_name)

    def get_classy_state(self, deep_copy=False):
        """
        Return the model state (trunk + heads) to checkpoint.

        We call this on the state.base_model which is not wrapped with DDP.
        get the model state_dict to checkpoint
        """

        if isinstance(self.trunk, FSDP):
            trunk_state_dict = self.trunk.local_state_dict()
            trunk_metadata_dict = self.trunk.local_metadata_dict()
        elif isinstance(self.trunk, FeatureExtractorModel) and isinstance(
            self.trunk.base_model, FSDP
        ):
            trunk_state_dict = self.trunk.base_model.local_state_dict()
            trunk_metadata_dict = self.trunk.base_model.local_metadata_dict()
        else:
            trunk_state_dict = self.trunk.state_dict()
            trunk_metadata_dict = {}

        if any(isinstance(head, FSDP) for head in self.heads):
            heads_state_dict = [
                head.local_state_dict() if isinstance(head, FSDP) else head.state_dict()
                for head in self.heads
            ]
            heads_metadata_dict = [
                head.local_metadata_dict() if isinstance(head, FSDP) else {}
                for head in self.heads
            ]
        else:
            heads_state_dict = self.heads.state_dict()
            heads_metadata_dict = []

        model_state_dict = {
            "model": {"trunk": trunk_state_dict, "heads": heads_state_dict},
            "meta": {"trunk": trunk_metadata_dict, "heads": heads_metadata_dict},
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        # ................... DEBUG ...............................
        # from vissl.utils.checkpoint import print_state_dict_shapes

        # print_state_dict_shapes(trunk_state_dict)
        # print_state_dict_shapes_shapes(heads_state_dict)
        return model_state_dict

    def set_classy_state(self, state, strict=True):
        """
        Initialize the model trunk and head from the state dictionary.

        We call this on the state.base_model which is not wrapped with DDP.
        load the model from checkpoint.
        """
        from vissl.utils.checkpoint import print_loaded_dict_info

        # Loading the trunk
        logging.info(f"Rank {self.local_rank}: Loading Trunk state dict....")
        if isinstance(self.trunk, FSDP):
            self.trunk.load_local_state_dict(state["model"]["trunk"])
            fsdp_recursive_reset_lazy_init(self.trunk)
        elif isinstance(self.trunk, FeatureExtractorModel) and isinstance(
            self.trunk.base_model, FSDP
        ):
            self.trunk.base_model.load_local_state_dict(state["model"]["trunk"])
            fsdp_recursive_reset_lazy_init(self.trunk.base_model)
        else:
            self.trunk.load_state_dict(state["model"]["trunk"])

        # Loading the head
        logging.info(f"Rank {self.local_rank}: Loading Heads state dict....")
        if any(isinstance(head, FSDP) for head in self.heads):
            for i, head in enumerate(self.heads):
                if isinstance(head, FSDP):
                    head.load_local_state_dict(state["model"]["heads"][i])
                    fsdp_recursive_reset_lazy_init(head)
                else:
                    self.head.load_state_dict(state["model"]["heads"][i])
        else:
            # sometimes, we want to load the partial head only, so strict=False
            self.heads.load_state_dict(state["model"]["heads"], strict=False)

        # Print debug information about layers loaded.
        #
        # Get the model state dict original (if FSDP, calling it on all ranks.)
        logging.info(f"Rank {self.local_rank}: Model state dict loaded!")
        if isinstance(self.trunk, FSDP) or any(
            isinstance(head, FSDP) for head in self.heads
        ):
            # TODO (Quentin) - log the weights of the loaded shard
            return

        if self.local_rank == 0:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
            model_state_dict = {}
            model_state_dict.update(trunk_state_dict)
            model_state_dict.update(heads_state_dict)

            # get the checkpoint state dict
            checkpoint_state_dict = {}
            checkpoint_state_dict.update(state["model"]["trunk"])
            checkpoint_state_dict.update(state["model"]["heads"])
            params_from_file = self.model_config["WEIGHTS_INIT"]
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            print_loaded_dict_info(
                model_state_dict,
                checkpoint_state_dict,
                skip_layers=skip_layers,
                model_config=self.model_config,
            )

    def init_model_from_weights_params_file(
        self, config: AttrDict, checkpoint: Dict[str, Any], strict: bool = False
    ):
        """
        We initialize the weights from this checkpoint. However, we don't care
        about the other metadata like iteration number etc.
        So the method only reads the state_dict
        """

        # Specific case for FSDP trunks:
        # - models have to be created with VISSL
        # - checkpoints have to be created with VISSL
        if isinstance(self.trunk, FeatureExtractorModel) and isinstance(
            self.trunk.base_model, FSDP
        ):
            # Linear evaluation / extraction from FSDP models:
            # - load the trunk (complete strict load)
            # - load the head (optional and partial load supported)
            logging.info("Loading FSDP trunk in extraction mode")
            CheckpointLoader.init_fsdp_model_from_weights(
                self.trunk.base_model,
                checkpoint,
                weights_path=["classy_state_dict", "base_model", "model", "trunk"],
            )
            fsdp_recursive_reset_lazy_init(self.trunk.base_model)
            if should_init_head_weights(config.MODEL):
                self._init_fsdp_model_heads_from_weights_params_file(checkpoint)

        elif isinstance(self.trunk, FSDP):
            # Fine-tuning of FSDP models:
            # - load the trunk (complete strict load)
            # - load the head (optional and partial load supported)
            logging.info("Loading FSDP trunk")
            CheckpointLoader.init_fsdp_model_from_weights(
                self.trunk,
                checkpoint,
                weights_path=["classy_state_dict", "base_model", "model", "trunk"],
            )
            fsdp_recursive_reset_lazy_init(self.trunk)
            if should_init_head_weights(config.MODEL):
                self._init_fsdp_model_heads_from_weights_params_file(checkpoint)

        # General case: support for multiple format of checkpoint
        else:
            params_from_file = config["MODEL"]["WEIGHTS_INIT"]
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            replace_prefix = params_from_file.get("REMOVE_PREFIX", None)
            append_prefix = params_from_file.get("APPEND_PREFIX", None)
            state_dict_key_name = params_from_file.get("STATE_DICT_KEY_NAME", None)
            init_model_from_consolidated_weights(
                config,
                self,
                checkpoint,
                state_dict_key_name=state_dict_key_name,
                skip_layers=skip_layers,
                replace_prefix=replace_prefix,
                append_prefix=append_prefix,
                strict=strict,
            )

    def _init_fsdp_model_heads_from_weights_params_file(
        self, checkpoint: Dict[str, Any]
    ):
        for i, head in enumerate(self.heads):
            logging.info(f"Loading FSDP head {i}")
            if isinstance(head, FSDP):
                CheckpointLoader.init_fsdp_model_from_weights(
                    head,
                    checkpoint,
                    weights_path=["classy_state_dict", "base_model", "model", "heads"],
                    strict=False,
                    head_index=i,
                )
                fsdp_recursive_reset_lazy_init(head)

    def is_clustering_model(self):
        """
        Whether or not the model is a clustering model
        (only supports SwAV for the moment)
        """
        return len(self.heads) == 1 and any(
            isinstance(module, SwAVPrototypesHead) for module in self.heads[0].modules()
        )

    @property
    def num_classes(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    def validate(self, dataset_output_shape):
        """
        Not implemented and not required
        """
        raise NotImplementedError
