# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

import pkg_resources
from setuptools import find_namespace_packages, find_packages, setup


def fetch_requirements():
    with pathlib.Path("requirements.txt").open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]
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
    author_email="vissl@fb.com",
    license="MIT",
    url="https://github.com/facebookresearch/vissl",
    description="VISSL is an extensible, modular and scalable library for "
    "SOTA Self-Supervised Learning with images.",
    packages=packages,
    install_requires=fetch_requirements(),
    include_package_data=True,
    python_requires=">=3.6.2",
    extras_require={
        "dev": [
            # "click==8.0.4",
            "black==22.3.0",
            # "black==19.3b0",
            "sphinx",
            "isort==5.7.0",
            "flake8==3.8.1",
            "flake8-bugbear",
            "flake8-comprehensions",
            "pre-commit",
            "nbconvert",
            "bs4",
            "faiss-gpu",
            "pycocotools>=2.0.1",
            "tensorboard>=1.15",
        ]
    },
)
