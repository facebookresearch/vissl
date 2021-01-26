# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import shutil
import subprocess

os.environ["SOURCE_ROOT_DIR"] = "/inside/apex"
os.environ["CONDA_CPUONLY_FEATURE"] = ""

python_versions = ["3.6", "3.7", "3.8", "3.9"]
# The CUDA versions which have pytorch conda packages available for linux for each
# version of pytorch.
# Pytorch 1.4 also supports cuda 10.0 but we no longer build for cuda 10.0 at all.
# cu92 did work but may not be in the container any more
CONDA_CUDA_VERSIONS = {
    "1.4": ["cu92", "cu101"],
    "1.5.0": ["cu92", "cu101", "cu102"],
    "1.5.1": ["cu92", "cu101", "cu102"],
    "1.6.0": ["cu92", "cu101", "cu102"],
    "1.7.0": ["cu101", "cu102", "cu110"],
    "1.7.1": ["cu101", "cu102", "cu110"],
}

CUDA_HOMES = {
    "cu110": "/usr/local/cuda-11.0/",
    "cu102": "/usr/local/cuda-10.2/",
    "cu101": "/usr/local/cuda-10.1/",
    "cu100": "/usr/local/cuda-10.0/",
    "cu92": "/usr/local/cuda-9.2/",
}
CUDATOOLKIT_CONSTRAINTS = {
    "cu110": "- cudatoolkit >=11.0,<11.1 # [not osx]",
    "cu102": "- cudatoolkit >=10.2,<10.3 # [not osx]",
    "cu101": "- cudatoolkit >=10.1,<10.2 # [not osx]",
    "cu100": "- cudatoolkit >=10.0,<10.1 # [not osx]",
    "cu92": "- cudatoolkit >=9.2,<9.3 # [not osx]",
}


def pytorch_versions_for_python(python_version):
    if python_version in ["3.6", "3.7", "3.8"]:
        return list(CONDA_CUDA_VERSIONS)
    pytorch_without_py39 = ["1.4", "1.5.0", "1.5.1", "1.6.0", "1.7.0"]
    return [i for i in CONDA_CUDA_VERSIONS if i not in pytorch_without_py39]


VERSION = "0.0"
# Uncomment this to use the official version number
# VERSION=$(python -c "exec(open('${script_dir}/apex/amp/__init__.py').read()); print(__version__)")
os.environ["BUILD_VERSION"] = VERSION

for python_version in python_versions:
    python_version_nodot = python_version.replace(".", "")
    os.environ["PYTHON_VERSION"] = python_version
    for ptv in pytorch_versions_for_python(python_version):
        os.environ["PYTORCH_VERSION"] = ptv
        ptv_nodot = ptv.replace(".", "")
        os.environ["PYTORCH_VERSION_NODOT"] = ptv_nodot
        os.environ["CONDA_PYTORCH_BUILD_CONSTRAINT"] = "- pytorch==" + ptv
        os.environ["CONDA_PYTORCH_CONSTRAINT"] = "- pytorch==" + ptv
        for cuv in CONDA_CUDA_VERSIONS[ptv]:
            os.environ["CU_VERSION"] = cuv
            os.environ["CUDA_HOME"] = CUDA_HOMES[cuv]
            os.environ["CONDA_CUDATOOLKIT_CONSTRAINT"] = CUDATOOLKIT_CONSTRAINTS[cuv]

            print()
            print("python", python_version, "pytorch", ptv, "cuda", cuv, flush=True)
            args = [
                "conda",
                "build",
                "-c",
                "pytorch",
                "-c",
                "defaults",
                "--no-anaconda-upload",
                "--python",
                python_version,
                "inside/packaging/apex",
            ]
            if python_version == "3.9":
                args.insert(4, "conda-forge")
                args.insert(4, "-c")
            subprocess.check_call(args)
            file = f"/opt/conda/conda-bld/linux-64/apex-{VERSION}-py{python_version_nodot}_{cuv}_pyt{ptv_nodot}.tar.bz2"
            shutil.copy(file, "inside/packaging")

print("DONE")
