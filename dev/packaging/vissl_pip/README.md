## Building VISSL pip packages

1. Make sure this checkout is on a filesystem which docker can
use - e.g. not NFS.

2. You may want to `docker pull pytorch/conda-cuda:latest`.

3. Run `bash go.sh` in this directory.

4. You can upload the packages to s3, along with mini html files
which enable them to be used, with `bash after.sh`.

In particular, if you are in a jupyter/colab notebook you can
then install using these wheels with the following series of
commands.


```
import sys
tag=f"py3.{sys.version_info.minor}"
!pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/{tag}/download.html
```