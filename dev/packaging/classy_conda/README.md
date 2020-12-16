## Building ClassyVision conda Packages

This does not use anything outside this directory.
It is a standalone set of code to build conda packages
for ClassyVision based on its 0.5 release.

1. Ensure you are have a conda environment loaded which has anaconda-client
and conda-build installed.

2. From this directory run
`bash build_all_conda.sh` to build the packages.

3. You can upload the packages to anaconda cloud by pasting your conda API
token over the word `redacted` in `upload.sh` and then running
`bash upload.sh` from this directory.
