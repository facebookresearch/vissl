## Building Nvidia apex conda Packages

This does not use anything outside this directory.
It is a standalone set of code to build conda packages
for apex.

1. Make sure this directory is on a filesystem which docker can
use - e.g. not NFS.

2. Go into the `inside` directory and clone apex.

3. You may want to `docker pull pytorch/conda-cuda:latest`.

4. Run `sudo bash go.sh` in this directory. This takes ages
and writes packages to `inside/packaging`.

5. You can upload the packages to anaconda cloud with
`bash upload.sh`.
