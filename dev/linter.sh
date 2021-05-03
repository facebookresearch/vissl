#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Run this script at project root by "./dev/linter.sh" before you commit

{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  python -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")
echo "$DIR"

echo "Running isort..."
isort -c --sp "${DIR}" "${DIR}"

echo "Running black..."
black "${DIR}"

echo "Running flake8..."
flake8 --max-line-length 88 --ignore E501,E203,E266,W503,E741 "${DIR}"
