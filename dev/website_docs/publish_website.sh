#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build and push updated VISSL site."
  echo ""
  exit 1
}

# Current directory (needed for cleanup later)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit
echo "${WORK_DIR}"

# Clone both master & gh-pages branches
git clone https://github.com/facebookresearch/vissl.git vissl-master
git clone --branch gh-pages https://github.com/facebookresearch/vissl.git vissl-gh-pages

cd vissl-master/website || exit

# Build site, tagged with "latest" version; baseUrl set to /versions/latest/
yarn
yarn run build

cd .. || exit
./dev/website_docs/build_website.sh -b

cd "${WORK_DIR}" || exit
rm -rf vissl-gh-pages/*
touch vissl-gh-pages/CNAME
echo "vissl.ai" > vissl-gh-pages/CNAME
mv vissl-master/website/build/vissl/* vissl-gh-pages/

cd vissl-gh-pages || exit
git add .
git commit -m 'Update latest version of site'
git push

# Clean up
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
