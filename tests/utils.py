# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import sys
from typing import Any, List

import pkg_resources
from omegaconf import OmegaConf
from vissl.utils.hydra_config import compose_hydra_configuration


logger = logging.getLogger("vissl")


# List all the config files, used to generate the unit tests on the fly
def list_config_files(dir_path, exclude_folders):
    resource_name = "configs"
    assert pkg_resources.resource_isdir(resource_name, dir_path)
    all_items = pkg_resources.resource_listdir(resource_name, dir_path)
    config_files = []

    def valid_file(filename):
        if not filename.endswith("yaml"):
            return False
        if exclude_folders and any(x in filename for x in exclude_folders):
            return False
        return True

    for item in all_items:
        subpath = f"{dir_path}/{item}"
        if pkg_resources.resource_isdir(resource_name, subpath):
            # Recursively test all the tree
            config_files.extend(list_config_files(subpath, exclude_folders))
        if valid_file(subpath):
            # If valid leaf, return the config file
            config_files.append(subpath)
    return config_files


def create_valid_input(input_list):
    out_list = []
    for item in input_list:
        out_list.append(re.sub("config/", "config=", item))
    return out_list


# we skip object detection configs since they are for detectron2 codebase
BENCHMARK_CONFIGS = create_valid_input(
    list_config_files("config/benchmark", exclude_folders=["object_detection"])
)

PRETRAIN_CONFIGS = create_valid_input(
    list_config_files("config/pretrain", exclude_folders=None)
)

INTEGRATION_TEST_CONFIGS = create_valid_input(
    list_config_files("config/test/integration_test", exclude_folders=None)
)

ROOT_CONFIGS = create_valid_input(
    list_config_files(
        "config", exclude_folders=["models", "optimization", "object_detection"]
    )
)

ROOT_OSS_CONFIGS = create_valid_input(
    list_config_files(
        "config", exclude_folders=["models", "optimization", "object_detection", "fb"]
    )
)

# configs that require loss optimization and hence trainable
ROOT_LOSS_CONFIGS = create_valid_input(
    list_config_files(
        "config",
        exclude_folders=[
            "models",
            "optimization",
            "object_detection",
            "nearest_neighbor",
            "feature_extraction",
            "fb",
            "test/transforms",
        ],
    )
)


UNIT_TEST_CONFIGS = create_valid_input(
    list_config_files("config/test/cpu_test", exclude_folders=None)
)


class SSLHydraConfig(object):
    def __init__(self, overrides: List[Any] = None):
        self.overrides = []
        if overrides is not None and len(overrides) > 0:
            self.overrides.extend(overrides)
        cfg = compose_hydra_configuration(self.overrides)
        self.default_cfg = cfg

    @classmethod
    def from_configs(cls, config_files: List[Any] = None):
        return cls(config_files)

    def override(self, config_files: List[Any]):
        sys.argv = config_files
        cli_conf = OmegaConf.from_cli(config_files)
        self.default_cfg = OmegaConf.merge(self.default_cfg, cli_conf)
