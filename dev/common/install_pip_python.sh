#!/bin/bash

set -ex

# Install common dependencies
apt-get update
apt-get install -y --no-install-recommends python3-dev python3-setuptools
ln -sv /usr/bin/python3 /usr/bin/python

# Install pip from source. The python-pip package on Ubuntu Trusty is old
# and upon install numpy doesn't use the binary distribution, and fails to compile it from source.
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
rm get-pip.py

# print the python
which python
python --version

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
