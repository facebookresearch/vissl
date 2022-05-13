# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vissl.config import AttrDict


@dataclass()
class MultipleInputMapping:
    """Configuration used to map multiple inputs to different part of the model

    Assumptions:
    - the same trunk is used to extract the features (but we can extract the features
      at different levels for each of the inputs)
    - by default, each input correspond to a different head (but this can be changed)

    Supported input formats:

        List (Legacy format):
        ["input_key", [list of features heads is applied on]]

        Dict (Flexible format):
        ["input_key", {"feat_names": [], "head_index": 0, "additional_keys":["mask"]}]

    """

    input_keys: List[str] = field(default_factory=list)
    feat_names: Dict[str, List[str]] = field(default_factory=dict)
    head_index: Dict[str, int] = field(default_factory=dict)
    additional_keys: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, model_config: AttrDict) -> Optional["MultipleInputMapping"]:
        if len(model_config.MULTI_INPUT_HEAD_MAPPING) == 0:
            return None

        mapping = MultipleInputMapping()
        num_heads = len(model_config.HEAD.PARAMS)
        for idx, mapping_config in enumerate(model_config.MULTI_INPUT_HEAD_MAPPING):
            mapping._parse_entry(idx, mapping_config, num_heads)
        return mapping

    def _parse_entry(self, idx: int, mapping_config: AttrDict, num_heads: int):
        assert len(mapping_config) == 2, "Invalid format"
        if isinstance(mapping_config[1], list):
            input_key = self._add_key(mapping_config[0])
            self.feat_names[input_key] = mapping_config[1]
            self.head_index[input_key] = min(idx, num_heads - 1)
            self.additional_keys[input_key] = []
        else:
            input_key = self._add_key(mapping_config[0])
            self.feat_names[input_key] = mapping_config[1]["feat_names"]
            self.head_index[input_key] = mapping_config[1].get(
                "head_index", min(idx, num_heads - 1)
            )
            self.additional_keys[input_key] = mapping_config[1].get(
                "additional_keys", []
            )

    def _add_key(self, key: str):
        if key in self.input_keys:
            raise ValueError(
                f"duplicate key {key} specified for MODEL.MULTI_INPUT_HEAD_MAPPING."
            )
        self.input_keys.append(key)
        return key
