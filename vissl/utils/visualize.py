# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def matplotlib_figure_to_image(fig):
    """
    Convert a matplotlib figure to an image in RGB format, for instance
    to save it on disk
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
