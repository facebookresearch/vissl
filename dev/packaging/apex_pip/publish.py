# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path


def write_html_wrappers():
    html = """
    <a href="$">$</a><br>
    """

    output = Path("inside/output")
    for directory in sorted(output.iterdir()):
        files = list(directory.glob("*.whl"))
        assert len(files) == 1, files
        [wheel] = files

        this_html = html.replace("$", wheel.name)
        (directory / "download.html").write_text(this_html)


def fs3cmd(args):
    """
    This function mimics the bash command fs3cmd available in the
    fairusers_aws module on the FAIR cluster.
    Works on H2.
    Not tested on H1 - this is a guess based on the definition in H2.
    """
    os.environ["FAIR_CLUSTER_NAME"] = os.environ["FAIR_ENV_CLUSTER"].lower()
    subprocess.check_call(["/public/apps/fairusers_aws/bin/fs3cmd"] + args)


def to_aws():
    dest = "s3://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/"

    output = Path("inside/output")
    for directory in sorted(output.iterdir()):
        for file in directory.iterdir():
            print(file)
            fs3cmd(["put", str(file), dest + str(file.relative_to(output))])


if __name__ == "__main__":
    write_html_wrappers()
    to_aws()
    # fs3cmd(["--help"])
