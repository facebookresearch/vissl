# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py,
pulished under an Apache License 2.0, with modifications by Matthew Leavitt (
ito@fb.com; matthew.l.leavitt@gmail.com). Modifications are described here and
notated where present in the code.

Modifications:
-Removed AugMix functionality.
-Replaced AutoAugment and RandAugment classes, which are no longer passed a
single parameter string that needs to be parsed, but instead individual,
named parameters.

COMMENT FROM ORIGINAL:
AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and
improvements and does not include any of the search code. AA and RA
Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection
    https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation...
    https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and
    Uncertainty https://arxiv.org/abs/1912.02781

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import re

import numpy as np
import PIL
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image, ImageEnhance, ImageOps


# TODO: Uncomment in future update when calling via ClassyVision
# from classy_vision.dataset.transforms.timm_autoaugment import \
#     _RAND_TRANSFORMS, _RAND_INCREASING_TRANSFORMS, rand_augment_ops, \
#     _HPARAMS_DEFAULT, _select_rand_weights, auto_augment_policy


# TODO: Delete in future update when calling via ClassyVision
_PIL_VER = tuple(int(x) for x in PIL.__version__.split(".")[:2])

# TODO: Delete in future update when calling via ClassyVision
_FILL = (128, 128, 128)

# TODO: Delete in future update when calling via ClassyVision
# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0

# TODO: Delete in future update when calling via ClassyVision
_HPARAMS_DEFAULT = {"translate_const": 250, "img_mean": _FILL}

# TODO: Delete in future update when calling via ClassyVision
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


# Modification/Addition
@register_transform("RandAugment", bypass_checks=True)
class RandAugment(ClassyTransform):
    """
    Create a RandAugment transform.
    :param magnitude: integer magnitude of rand augment
    :param magnitude_std: standard deviation of magnitude. If > 0, introduces
    random variability in the augmentation magnitude.
    :param num_layers: integer number of transforms
    :param increasing_severity: boolean that indicates whether to use
    augmentations that increase severity w/ increasing magnitude. Some
    augmentations do this by default.
    :param weight_choice: Index of pre-determined probability distribution
    over augmentations. Currently only one such distribution available (i.e.
    no valid values other than 0 or None), unclear if beneficial. Default =
    None.
    """

    def __init__(
        self,
        magnitude=10,
        magnitude_std=0,
        num_layers=2,
        increasing_severity=False,
        weight_choice=None,
        **kwargs,
    ):
        hparams = kwargs
        hparams.update(_HPARAMS_DEFAULT)
        hparams["magnitude_std"] = magnitude_std
        if increasing_severity:
            transforms = _RAND_INCREASING_TRANSFORMS
        else:
            transforms = _RAND_TRANSFORMS
        self.num_layers = num_layers
        self.choice_weights = (
            None if weight_choice is None else _select_rand_weights(weight_choice)
        )
        self.ops = rand_augment_ops(
            magnitude=magnitude, hparams=hparams, transforms=transforms
        )

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img


# Modification/Addition
@register_transform("VisslAutoAugment")
class AutoAugment(ClassyTransform):
    """
    Create a AutoAugment transform. This autoaugment differs from the
    torchvision implementation by allowing variability in the augmentation
    intensity.
    ":param policy_name: String. One of 'v0', 'v0r', 'original', 'originalr'.
    One of a set of learned augmentation sequences.
    :param magnitude_std: standard deviation of magnitude. If > 0, introduces
    random variability in the augmentation magnitude.
    :kwargs: Other params for the AutoAugmentation scheme. See RandAugment
    class above, or AugmentOp class in ClassyVision. Probability and
    intensity are overwritten because they're determined by the learned
    AutoAugment policy.
    """

    def __init__(self, policy_name="v0", magnitude_std=0, **kwargs):
        hparams = kwargs
        hparams.update(_HPARAMS_DEFAULT)
        hparams["magnitude_std"] = magnitude_std
        self.policy = auto_augment_policy(policy_name, hparams=hparams)

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img


# TODO: Delete everything from here down in future update when calling via
# ClassyVision
# Everything from here down is copied directly from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0
    # increases the enhancement blend range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various
    # Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = {
            "fillcolor": hparams["img_mean"] if "img_mean" in hparams else _FILL,
            "resample": (
                hparams["interpolation"]
                if "interpolation" in hparams
                else _RANDOM_INTERPOLATION
            ),
        }

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = (
            self.level_fn(magnitude, self.hparams) if self.level_fn is not None else ()
        )
        return self.aug_fn(img, *level_args, **self.kwargs)


def auto_augment_policy_v0(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, cannot find a paper reference.
    policy = [
        [("Equalize", 0.8, 1), ("ShearY", 0.8, 4)],
        [("Color", 0.4, 9), ("Equalize", 0.6, 3)],
        [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
        [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
        [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
        [("Color", 0.2, 0), ("Equalize", 0.8, 8)],
        [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],
        [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
        [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
        [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],
        [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],
        [("Color", 0.4, 7), ("Equalize", 0.6, 0)],
        [("Posterize", 0.4, 6), ("AutoContrast", 0.4, 7)],
        [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
        [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
        [("Rotate", 1.0, 7), ("TranslateYRel", 0.8, 9)],
        [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
        [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
        [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
        [("Equalize", 0.8, 4), ("Equalize", 0.0, 8)],
        [("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)],
        [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
        [
            ("Posterize", 0.8, 2),
            ("Solarize", 0.6, 10),
        ],  # This results in black image with Tpu posterize
        [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
        [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0r(hparams):
    # ImageNet v0 policy from TPU EfficientNet impl, with variation of Posterize used
    # in Google research implementation (number of bits discarded increases with magnitude)
    policy = [
        [("Equalize", 0.8, 1), ("ShearY", 0.8, 4)],
        [("Color", 0.4, 9), ("Equalize", 0.6, 3)],
        [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
        [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
        [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
        [("Color", 0.2, 0), ("Equalize", 0.8, 8)],
        [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],
        [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
        [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
        [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],
        [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],
        [("Color", 0.4, 7), ("Equalize", 0.6, 0)],
        [("PosterizeIncreasing", 0.4, 6), ("AutoContrast", 0.4, 7)],
        [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
        [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
        [("Rotate", 1.0, 7), ("TranslateYRel", 0.8, 9)],
        [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
        [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
        [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
        [("Equalize", 0.8, 4), ("Equalize", 0.0, 8)],
        [("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)],
        [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 2), ("Solarize", 0.6, 10)],
        [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
        [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_original(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501
    policy = [
        [("PosterizeOriginal", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeOriginal", 0.6, 7), ("PosterizeOriginal", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeOriginal", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeOriginal", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_originalr(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501 with research posterize variation
    policy = [
        [("PosterizeIncreasing", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeIncreasing", 0.6, 7), ("PosterizeIncreasing", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeIncreasing", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name="v0", hparams=None):
    hparams = hparams or _HPARAMS_DEFAULT
    if name == "original":
        return auto_augment_policy_original(hparams)
    elif name == "originalr":
        return auto_augment_policy_originalr(hparams)
    elif name == "v0":
        return auto_augment_policy_v0(hparams)
    elif name == "v0r":
        return auto_augment_policy_v0r(hparams)
    else:
        assert AssertionError, "Unknown AA policy (%s)" % name


def auto_augment_transform(config_str, hparams):
    """
    Create a AutoAugment transform
    :param config_str: String defining configuration of auto augmentation.
    Consists of multiple sections separated by dashes ('-'). The first
    section defines the AutoAugment policy (one of 'v0', 'v0r', 'original',
    'originalr').
    The remaining sections, not order sepecific determine
        'mstd' -  float std deviation of magnitude noise applied
    Ex 'original-mstd0.5' results in AutoAugment with original policy, magnitude_std 0.5
    :param hparams: Other hparams (kwargs) for the AutoAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    config = config_str.split("-")
    policy_name = config[0]
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        else:
            assert AssertionError, "Unknown AutoAugment config section"
    aa_policy = auto_augment_policy(policy_name, hparams=hparams)
    return AutoAugment(aa_policy)


_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
        for name in transforms
    ]
