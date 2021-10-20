## Building Nvidia apex pip Packages

This does not use anything outside this directory.
It is a standalone set of code to build wheels
for apex.

1. Make sure this directory is on a filesystem which docker can
use - e.g. not NFS. If you are using a local hard drive there is
nothing to do here.

2. Go into the `inside` directory and clone apex with
`git clone https://github.com/NVIDIA/apex.git`.

Move to the appropriate commit.
`git checkout 9ce0a10fb6c2537ef6a59f27b7875e32a9e9b8b8`.

3. You may want to `docker pull pytorch/conda-cuda:latest`.

4. Run `bash go.sh` in this directory. This takes ages
and writes packages to `inside/output`.

5. You can upload the packages to s3, along with basic html files
which enable them to be used, with `bash after.sh`.

In particular, if you are in a jupyter/colab notebook you can
then install using these wheels with the following series of
commands.

```
import sys
import torch
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{torch.__version__[0:5:2]}"
])
!pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/{version_str}/download.html apex
```
