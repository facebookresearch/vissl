## Building VISSL pip packages

1. Make sure this checkout is on a filesystem which docker can
use - e.g. not NFS.

2. You may want to `docker pull pytorch/conda-cuda:latest`.

4. Run `bash go.sh` in this directory. This takes ages
and writes packages to `output` in this directory.

5. You can upload the packages to s3, along with mini html files
which enable them to be used, with `bash after.sh`.


In particular, if you are in a jupyter/colab notebook you can
then install using these wheels with the following series of
commands.


```
!pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/download.html
```