# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pprint
import sys
from typing import Any, List

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


def hydra_main(overrides: List[Any]):
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    print(pprint.pformat(config))


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)
