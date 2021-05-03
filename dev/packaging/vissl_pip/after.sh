#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -ex

sudo chown -R "$USER" output
python publish.py
bash to_pypi.sh
