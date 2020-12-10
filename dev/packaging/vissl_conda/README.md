## Building VISSL conda Packages

1. Ensure you are have a conda environment loaded which has anaconda-client
and conda-build installed.

2. From the root of the repository run
`bash dev/packaging/vissl_conda/build_all_conda.sh` to build the packages.

3. You can upload the packages to anaconda cloud by running
`bash upload.sh` from this directory.
