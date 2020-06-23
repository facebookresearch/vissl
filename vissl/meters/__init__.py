#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

# automatically import any Python files in the meters/ directory
import_all_modules(FILE_ROOT, "vissl.meters")
