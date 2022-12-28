# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "vissl.optimizers.param_groups")
