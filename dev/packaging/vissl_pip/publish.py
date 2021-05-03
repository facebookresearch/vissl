# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path


dest = "s3://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/"

# we build on python3.6 but it works for 3.7, 3.8 and 3.9 too
output = Path("output/py3.6")


def fs3cmd(args, allow_failure=False):
    """
    This function returns the args for subprocess to mimic the bash command
    fs3cmd available in the fairusers_aws module on the FAIR cluster.
    Works on H2.
    Not tested on H1 - this is a guess based on the definition in H2.
    """
    os.environ["FAIR_CLUSTER_NAME"] = os.environ["FAIR_ENV_CLUSTER"].lower()
    cmd_args = ["/public/apps/fairusers_aws/bin/fs3cmd"] + args
    return cmd_args


def fs3_exists(path):
    """
    Returns True if the path exists inside dest on S3.
    In fact, will also return True if there is a file which has the given
    path as a prefix, but we are careful about this.
    """
    out = subprocess.check_output(fs3cmd(["ls", path]))
    return len(out) != 0


def get_html_wrappers():
    output_wrapper = output / "download.html"
    assert not output_wrapper.exists()
    dest_wrapper = dest + "download.html"
    if fs3_exists(dest_wrapper):
        fs3cmd(["get", dest_wrapper, str(output_wrapper)])


def write_html_wrappers():
    html = """
    <a href="$">$</a><br>
    """

    files = list(output.glob("*.whl"))
    assert len(files) == 1, files
    [wheel] = files

    this_html = html.replace("$", wheel.name)
    output_wrapper = output / "download.html"
    if output_wrapper.exists():
        contents = output_wrapper.read_text()
        if this_html not in contents:
            with open(output_wrapper, "a") as f:
                f.write(this_html)
    else:
        output_wrapper.write_text(this_html)


def to_aws():
    for file in output.iterdir():
        print(file)
        subprocess.check_call(
            fs3cmd(["put", str(file), dest + str(file.relative_to(output))])
        )


if __name__ == "__main__":
    get_html_wrappers()
    write_html_wrappers()
    to_aws()
    # subprocess.check_call(fs3cmd(["get", "--help"]))
