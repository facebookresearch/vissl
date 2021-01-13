# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

# automatically import any Python files in the param_scheduler/ directory
import_all_modules(FILE_ROOT, "vissl.optimizers.param_scheduler")
