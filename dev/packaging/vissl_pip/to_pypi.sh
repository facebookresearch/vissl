#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -e

TOKEN=redacted

twine  upload --verbose --username __token__ --password $TOKEN output/py3.6/*.whl
