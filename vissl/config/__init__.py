# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from vissl.config.attr_dict import AttrDict


# When to do version bump:
#   - version bump is NOT required if the new keys are being added to the defaults.yaml
#   - version bump IS required when the existing keys are being changed leading to
#     potential backward incompatibility. Example: deprecation of existing keys,
#     existing keys behavior changes etc.
# To bump version:
#     1. Increment LATEST_CFG_VERSION in this file
#     2. Increment the VERSION in defaults.yaml file.
#     2. Add a proper version converter in this file.


LATEST_CFG_VERSION = 1


def check_cfg_version(cfg: AttrDict):
    """
    Check the config version

    Inputs:
        cfg (AttrDict): the config to be checked
    """

    cfg_version = cfg.VERSION

    # if the versions already match, no upgrade is required.
    if cfg_version == LATEST_CFG_VERSION:
        logging.info(f"Provided Config has latest version: {LATEST_CFG_VERSION}")
        return

    # make sure the config is valid: the provided config version must be
    # less than or equal to VISSL latest config version.
    assert cfg_version <= LATEST_CFG_VERSION, (
        f"Provided Config version is: {cfg_version}. "
        f"VISSL latest config version is: {LATEST_CFG_VERSION}. "
        "Config version is not supported in VISSL. Max supported "
        f"version is {LATEST_CFG_VERSION}"
    )

    # TODO: support version conversion when needed
    if cfg_version < LATEST_CFG_VERSION:
        raise RuntimeError(
            "Please upgrade your config to latest VISSL configuration setup. "
            "Follow the change log to see what has changed in configs and how to upgrade."
        )


__all__ = ["LATEST_CFG_VERSION", "check_cfg_version"]
