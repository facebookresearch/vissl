## Building VISSL pip packages

1. Make sure this checkout is on a filesystem which docker can
use - e.g. not NFS. If you are using a local hard drive there is
nothing to do here.

2. You may want to `docker pull pytorch/conda-cuda:latest`.

3. Run `bash go.sh` in this directory.

4. You can upload the packages to s3, along with basic html files
which enable them to be used, and to PyPI, with `bash after.sh`.
First paste your PyPI API token in place of the word `redacted`
in `to_pypi.sh`.

In particular, if you are in a jupyter/colab notebook you can
then install using these wheels with the following.

```
!pip install vissl -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/download.html
```

5. In the test directory here you can find a tool to run the tests against
the PyPI upload.
