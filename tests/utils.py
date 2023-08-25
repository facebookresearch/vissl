# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import sys
from typing import Any, Callable, List

import pkg_resources
from omegaconf import OmegaConf
from vissl.utils.hydra_config import compose_hydra_configuration


logger = logging.getLogger("vissl")


# List all the config files, used to generate the unit tests on the fly
def list_config_files(
    dir_path: str, is_valid_file: Callable[[str], bool] = lambda x: True
):
    """
    Recursively list all YAML files under the folder "configs / dir_path"
    excluding all paths in which one of the terms in "exclude_folders" appear
    """
    resource_name = "configs"
    assert pkg_resources.resource_isdir(resource_name, dir_path)
    all_items = pkg_resources.resource_listdir(resource_name, dir_path)
    config_files = []

    for item in all_items:
        subpath = f"{dir_path}/{item}"
        if pkg_resources.resource_isdir(resource_name, subpath):
            config_files.extend(list_config_files(subpath, is_valid_file))
        elif subpath.endswith(".yaml") and is_valid_file(subpath):
            config_files.append(subpath)
    return config_files


def exclude_folders(folders: List[str]) -> Callable[[str], bool]:
    """
    Predicate for list_config_files allowing to list all files
    but the ones in specific list of invalid folders
    """

    def is_path_allowed(path: str) -> bool:
        if exclude_folders and any(x in path for x in folders):
            return False
        return True

    return is_path_allowed


def only_benchmark_models(path: str) -> bool:
    """
    Predicate for list_config_files allowing to list all files
    that are benchmarks or models used in benchmarks
    """
    dir_path, file_name = os.path.split(path)
    return file_name.startswith("eval_") or dir_path.endswith("models")


def only_pretrain_models(path: str) -> bool:
    """
    Predicate for list_config_files allowing to list all files
    that are pre-training methods or models plugged in those methods
    """
    dir_path, _ = os.path.split(path)
    return dir_path.endswith("pretrain") or dir_path.endswith("models")


def create_valid_input(input_list: List[str]) -> List[str]:
    return [re.sub("config/", "config=", item) for item in input_list]


# we skip object detection configs since they are for detectron2 codebase
BENCHMARK_CONFIGS = create_valid_input(
    list_config_files(
        "config/benchmark",
        exclude_folders(
            [
                "object_detection",
                "datasets",
                "models",  # will be tested via BENCHMARK_MODEL_CONFIGS
            ]
        ),
    )
)

BENCHMARK_MODEL_CONFIGS = create_valid_input(
    list_config_files("config/benchmark", only_benchmark_models)
)

PRETRAIN_CONFIGS = create_valid_input(list_config_files("config/pretrain"))

PRETRAIN_MODEL_CONFIGS = create_valid_input(
    list_config_files("config/pretrain", only_pretrain_models)
)

INTEGRATION_TEST_CONFIGS = create_valid_input(
    list_config_files("config/test/integration_test")
)

# configs that require loss optimization and hence trainable
ROOT_LOSS_CONFIGS = create_valid_input(
    list_config_files(
        "config",
        exclude_folders(
            [
                "datasets",
                "fb",
                "feature_extraction",
                "models",
                "nearest_neighbor",
                "optimization",
                "object_detection",
                "transforms",
            ]
        ),
    )
)


UNIT_TEST_CONFIGS = create_valid_input(list_config_files("config/test/cpu_test"))


class SSLHydraConfig:
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
