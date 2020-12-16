## Building VISSL conda Packages

1. Ensure you are have a conda environment loaded which has anaconda-client
and conda-build installed.

2. If you want the testing to work, make sure you have cuda 10.1 working as
that is assumed by the testing phase. E.g. on the FAIR cluster
```
module purge
module load cuda/10.1
module load NCCL/2.7.6-1-cuda.10.1
```

3. From the root of the repository run
`bash dev/packaging/vissl_conda/build_all_conda.sh` to build the packages.

4. You can upload the packages to anaconda cloud by pasting your conda API
token over the word `redacted` in `upload.sh` and then running
`bash upload.sh` from this directory.
