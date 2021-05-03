# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from collections import defaultdict

import numpy as np
import PIL
import pkg_resources
import torch
import torchvision
from tabulate import tabulate


__all__ = ["collect_env_info"]


def collect_torch_env():
    """
    If torch is available, print the torch config.
    """
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_python_info(data):
    """
    Collect python version, numpy and pillow version and the system platform
    information
    """
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))
    data.append(("Pillow", PIL.__version__))
    return data


def collect_gpus_info(data):
    """
    If the cuda is available on the system, collect information about
    CUDA_HOME, TORCH_CUDA_ARCH_LIST, GPU names, count
    """
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    data.append(("GPU available", has_gpu))
    from torch.utils.cpp_extension import CUDA_HOME

    # gpus info and other env settings
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
        data.append(("CUDA_HOME", str(CUDA_HOME)))
        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list:
            data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    return data


def collect_dep_env(data):
    """
    Collection information about VISSL dependendices and their version
    torchvision, hydra, classy vision, tensorboard, apex
    """
    # torchvision
    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__)
                + " @"
                + os.path.dirname(torchvision.__file__),
            )
        )
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import hydra

        data.append(
            ("hydra", str(hydra.__version__) + " @" + os.path.dirname(hydra.__file__))
        )
    except ImportError:
        pass

    try:
        import classy_vision

        data.append(
            (
                "classy_vision",
                str(classy_vision.__version__)
                + " @"
                + os.path.dirname(classy_vision.__file__),
            )
        )
    except ImportError:
        pass

    try:
        import tensorboard

        data.append(("tensorboard", tensorboard.__version__))
    except ImportError:
        pass

    try:
        import apex  # noqa

        data.append(
            (
                "apex",
                str(pkg_resources.get_distribution("apex").version)
                + " @"
                + os.path.dirname(apex.__file__),
            )
        )
    except ImportError:
        data.append(("apex", "unknown"))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass

    return data


def collect_vissl_info(data):
    """
    Collect information about vissl version being used
    """
    try:
        import vissl  # noqa

        data.append(
            ("vissl", vissl.__version__ + " @" + os.path.dirname(vissl.__file__))
        )
    except ImportError:
        data.append(("vissl", "failed to import"))
    return data


def collect_cpu_info():
    """
    collect information about system cpus.
    """
    with os.popen("lscpu") as f:
        cpu_info = f.readlines()
    out_cpu_info = []
    for item in cpu_info:
        key = item.strip("\n").split(":")[0].strip()
        if key == "Flags":
            continue
        out_cpu_info.append((key, item.strip("\n").split(":")[1].strip()))
    out_cpu_info = tabulate(out_cpu_info)
    return out_cpu_info


def collect_env_info():
    """
    Collect information about user system including cuda, torch, gpus, vissl and its
    dependencies. Users are strongly recommended to run this script to collect
    information about information if they needed debugging help.
    """
    # common python info
    data = []
    data = collect_python_info(data)
    data = collect_vissl_info(data)
    data = collect_gpus_info(data)
    data = collect_dep_env(data)

    # append torch info
    torch_version = torch.__version__
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env() + "\n"
    env_str += "CPU info:\n"
    env_str += collect_cpu_info()
    return env_str


if __name__ == "__main__":
    try:
        import vissl  # noqa
    except ImportError:
        print(collect_env_info())
    else:
        from vissl.utils.collect_env import collect_env_info

        print(collect_env_info())
