# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


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
