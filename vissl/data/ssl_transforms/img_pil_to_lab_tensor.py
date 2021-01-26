# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict

import cv2
import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPil2LabTensor")
class ImgPil2LabTensor(ClassyTransform):
    """
    Convert a PIL image to LAB tensor of shape C x H x W
    This transform was proposed in Colorization - https://arxiv.org/abs/1603.08511

    The input image is PIL Image. We first convert it to tensor
    HWC which has channel order RGB. We then convert the RGB to BGR
    and use OpenCV to convert the image to LAB. The LAB image is
    8-bit image in range > L [0, 255], A [0, 255], B [0, 255]. We
    rescale it to: L [0, 100], A [-128, 127], B [-128, 127]

    The output is image torch tensor.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, image):
        img_tensor = np.array(image)
        # PIL image tensor is RGB. Convert to BGR
        img_bgr = img_tensor[:, :, ::-1]
        img_lab = self._convertbgr2lab(img_bgr.astype(np.uint8))
        # convert HWC -> CHW. The image is LAB.
        img_lab = np.transpose(img_lab, (2, 0, 1))
        # torch tensor output
        img_lab_tensor = torch.from_numpy(img_lab).float()
        return img_lab_tensor

    def _convertbgr2lab(self, img):
        # img is [0, 255] , HWC, BGR format, uint8 type
        assert len(img.shape) == 3, "Image should have dim H x W x 3"
        assert img.shape[2] == 3, "Image should have dim H x W x 3"
        assert img.dtype == np.uint8, "Image should be uint8 type"
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
        # L [0, 100], A [-128, 127], B [-128, 127]
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
        img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
        ############################ debugging ####################################
        # img_lab_bw = img_lab.copy()
        # img_lab_bw[:, :, 1:] = 0.0
        # img_lab_bgr = cv2.cvtColor(img_lab_bw, cv2.COLOR_Lab2BGR)
        # img_lab_bgr = img_lab_bgr.astype(np.float32)
        # img_lab_RGB = img_lab_bgr[:, :, [2, 1, 0]]        # BGR to RGB
        # img_lab_RGB = img_lab_RGB - np.min(img_lab_RGB)
        # img_lab_RGB /= np.max(img_lab_RGB) + np.finfo(np.float64).eps
        # plt.imshow(img_lab_RGB)
        # n = np.random.randint(0, 1000)
        # np.save(f"/tmp/lab{n}.npy", img_lab_bgr)
        # print("SAVED!!")
        ######################### debugging over ##################################
        return img_lab

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPil2LabTensor":
        """
        Instantiates ImgPil2LabTensor from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPil2LabTensor instance.
        """
        indices = config.get("indices", [])
        return cls(indices=indices)
