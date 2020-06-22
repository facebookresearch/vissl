#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import find_namespace_packages, find_packages, setup


packages = find_packages(exclude=("tests",)) + find_namespace_packages(
    include=["hydra_plugins.*"]
)

setup(
    name="ssl_framework",
    version="0.1.2",
    description="A toolkit for Self-Supervised Learning Research",
    packages=packages,
    include_package_data=True,
)
