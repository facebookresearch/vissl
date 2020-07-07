# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import PIL


"""
Generic Image Enhancements using PIL.
These image enhancements only modify the photometric properties
and do not alter the geometric properties of the image.
"""


class TransformObject(object):
    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        str = f"{self._get_name()}({self.root_transform})"
        return str


class RandomValueApplier(TransformObject):
    def __init__(
        self, min_v, max_v, root_transform, vtype="float", closed_interval=False
    ):
        """
        Applies a transform by sampling a random value between [min_v, max_v]
        Inputs
        - min_v (float or int): minimum value
        - max_v (float or int): maximum value
        - root_transform (transform object): transform that will be applied.
            must accept a value as input.
        - vtype (string): value type - either "float" or "int"
        - closed_interval (bool): sample from [min_v, max_v] (when True)
            or [min_v, max_v) when False
        """
        self.min_v = min_v
        self.max_v = max_v
        self.root_transform = root_transform
        self.vtype = vtype
        self.closed_interval = closed_interval
        if self.closed_interval:
            assert self.vtype == "int"

    def sample_value(self):
        if self.vtype == "float":
            v = np.random.uniform(low=self.min_v, high=self.max_v)
        elif self.vtype == "int":
            if self.closed_interval:
                v = np.random.randint(low=self.min_v, high=self.max_v + 1)
            else:
                v = np.random.randint(low=self.min_v, high=self.max_v)
        return v

    def __call__(self, img):
        v = self.sample_value()
        return self.root_transform(img, v)

    def __repr__(self):
        str = f"{self._get_name()}(min_v={self.min_v}, max_v={self.max_v}, root_transform={self.root_transform}, vtype={self.vtype}, closed_interval={self.closed_interval})"  # NOQA
        return str


def Sharpness(img, v):
    assert 0.1 <= v <= 1.9, f"{v} out of range [0.1, 1.9]"
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Solarize(img, v):
    assert 0 <= v <= 256, f"{v} out of range [0, 256]"
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):
    assert 4 <= v <= 8, f"{v} out of range [4, 8]"
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


class RandomSharpnessTransform(RandomValueApplier):
    def __init__(self, min_v=0.1, max_v=1.9, root_transform=Sharpness, vtype="float"):
        super(RandomSharpnessTransform, self).__init__(
            min_v, max_v, root_transform, vtype
        )


class RandomPosterizeTransform(RandomValueApplier):
    def __init__(self, min_v=4, max_v=8, root_transform=Posterize, vtype="int"):
        super(RandomPosterizeTransform, self).__init__(
            min_v, max_v, root_transform, vtype, True
        )


class RandomSolarizeTransform(RandomValueApplier):
    def __init__(self, min_v=0, max_v=256, root_transform=Solarize, vtype="int"):
        super(RandomSolarizeTransform, self).__init__(
            min_v, max_v, root_transform, vtype, True
        )


class AutoContrastTransform(TransformObject):
    def __init__(self):
        self.root_transform = AutoContrast

    def __call__(self, img):
        return self.root_transform(img, 0)
