## Building Nvidia apex conda Packages

This does not use anything outside this directory.
It is a standalone set of code to build conda packages
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
and writes packages to `inside/packaging`.

5. You can upload the packages to anaconda cloud by pasting your conda API
token over the word `redacted` in `upload.sh` and then running
`bash upload.sh` from this directory.
