#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

TOKEN=redacted

twine  upload --verbose --username __token__ --password $TOKEN output/py3.6/*.whl
