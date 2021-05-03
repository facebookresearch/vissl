#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run this script from the project root using `./dev/website_docs/build_website.sh`

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build VISSL documentation."
  echo ""
  echo "  -b   Build static version of documentation (otherwise start server)"
  echo ""
  exit 1
}

BUILD_STATIC=false

while getopts 'hb' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    b)
      BUILD_STATIC=true
      ;;
    *)
      usage
      ;;
  esac
done


echo "-----------------------------------"
echo "Building VISSL Docusaurus site"
echo "-----------------------------------"
cd website || exit
yarn
cd ..

echo "-----------------------------------"
echo "Generating tutorials"
echo "-----------------------------------"
cwd=$(pwd)
mkdir -p "website/_tutorials"
mkdir -p "website/static/files"
echo ${cwd}
python dev/website_docs/parse_tutorials.py --repo_dir "${cwd}"

cd website || exit

if [[ $BUILD_STATIC == true ]]; then
  echo "-----------------------------------"
  echo "Building static site"
  echo "-----------------------------------"
  yarn build
else
  echo "-----------------------------------"
  echo "Starting local server"
  echo "-----------------------------------"
  yarn start
fi
