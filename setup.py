#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pathlib
import pkg_resources
import sys

from setuptools import find_namespace_packages, find_packages, setup

WITH_APEX=os.getenv("WITH_APEX", "0")

print(f'######## USE_APEX: {WITH_APEX}')

def fetch_requirements():
    with pathlib.Path("requirements.txt").open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]
    if WITH_APEX == "1":
      install_requires.append('apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf')
      print("APEX PASSED")
    return install_requires


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "vissl", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


packages = find_packages(exclude=("tests",)) + find_namespace_packages(
    include=["hydra_plugins.*"]
)

setup(
    name="vissl",
    version=get_version(),
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/vissl",
    description="VISSL is an extensible, modular and scalable library for "
    "SOTA Self-Supervised Learning with images.",
    packages=packages,
    install_requires=fetch_requirements(),
    include_package_data=True,
    python_requires=">=3.6",
    extras_require={
        "dev": [
            "black==19.3b0",
            "sphinx",
            "isort",
            "flake8==3.8.1",
            "isort",
            "flake8-bugbear",
            "flake8-comprehensions",
            "pre-commit",
            "nbconvert",
        ]
    },
)
