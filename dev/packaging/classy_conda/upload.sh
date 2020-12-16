#!/usr/bin/bash

set -e

TOKEN=redacted
CHANNEL=vissl

retry () {
    # run a command, and try again if it fails
    $*  || (echo && sleep 8 && echo retrying && $*)
}

for file in out/linux-64/*.tar.bz2
do
    echo
    echo "${file}"
    retry anaconda --verbose -t "${TOKEN}" upload -u "${CHANNEL}" --force "${file}" --no-progress
done
