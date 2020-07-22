# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from vissl.data.ssl_transforms.img_pil_to_tensor import ImgToTensor


RAND_TENSOR = (torch.rand((224, 224, 3)) * 255).to(dtype=torch.uint8)
RAND_PIL = Image.fromarray(RAND_TENSOR.numpy())
RAND_NUMPY = np.asarray(RAND_PIL)


class TaskTest(unittest.TestCase):
    def test_to_tensor(self):
        # Check that ImgToTensor and torchvision.transforms.ToTensor
        # are mostly equivalent

        # PIL.Image
        a = ImgToTensor()(RAND_PIL)
        b = ToTensor()(RAND_PIL)
        self.assertTrue(torch.allclose(a, b))

        # Numpy array
        c = ImgToTensor()(RAND_NUMPY)
        d = ToTensor()(RAND_NUMPY)
        self.assertTrue(torch.allclose(c, d))
