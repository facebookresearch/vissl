#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

TOKEN=redacted

retry () {
    # run a command, and try again if it fails
    $*  || (echo && sleep 8 && echo retrying && $*)
}

for file in inside/packaging/*.bz2
do
    echo
    echo "${file}"
    retry anaconda --verbose -t "${TOKEN}" upload -u vissl --force "${file}" --no-progress
done
