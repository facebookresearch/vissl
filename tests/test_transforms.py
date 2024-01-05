# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from vissl.data.ssl_transforms.img_pil_to_multicrop import ImgPilToMultiCrop
from vissl.data.ssl_transforms.img_pil_to_tensor import ImgToTensor
from vissl.data.ssl_transforms.mnist_img_pil_to_rgb_mode import MNISTImgPil2RGB
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    augly_test,
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


RAND_TENSOR = (torch.rand((224, 224, 3)) * 255).to(dtype=torch.uint8)
RAND_PIL = Image.fromarray(RAND_TENSOR.numpy())
RAND_NUMPY = np.asarray(RAND_PIL)


class TestTransform(unittest.TestCase):
    def test_to_tensor(self) -> None:
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

    def test_img_pil_to_rgb_mode(self) -> None:
        one_channel_input = (torch.ones((28, 28)) * 255).to(dtype=torch.uint8)
        one_channel_input = Image.fromarray(one_channel_input.numpy(), mode="L")

        # Test without modifying the image size
        transform = Compose([MNISTImgPil2RGB.from_config({}), ToTensor()])
        output = transform(one_channel_input)
        assert output.shape == torch.Size([3, 28, 28])
        assert (
            output.sum().item() == 28 * 28 * 3
        ), "Background should be black, center is gray scale"

        # Test with modifying the image size (try the two valid formats)
        for size in [32, [32, 32]]:
            transform = Compose(
                [MNISTImgPil2RGB.from_config({"size": size, "box": [2, 2]}), ToTensor()]
            )
            output = transform(one_channel_input)
            assert (
                output.sum().item() == 28 * 28 * 3
            ), "Background should be black, center is gray scale"
            assert (
                output[:, 2:-2, 2:-2].sum().item() == 28 * 28 * 3
            ), "Paste should be in the middle"

    def test_img_pil_to_multicrop(self) -> None:
        torch.cuda.manual_seed(0)

        transform = ImgPilToMultiCrop(
            total_num_crops=8,
            num_crops=[2, 6],
            size_crops=[224, 96],
            crop_scales=[[0.14, 1], [0.05, 0.14]],
        )

        image = torch.randn(size=(3, 256, 256))
        image = Image.fromarray(image.numpy(), mode="RGB")
        crops = transform(image)
        self.assertEqual(8, len(crops))
        for crop in crops[:2]:
            self.assertEqual((224, 224), crop.size)
        for crop in crops[2:]:
            self.assertEqual((96, 96), crop.size)

    @augly_test()
    @gpu_test(gpu_count=2)
    def test_augly_transforms(self) -> None:
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_simclr",
                "+config/test/transforms=augly_transforms_example",
            ],
        )
        _, config = convert_to_attrdict(cfg)

        with in_temporary_directory() as _:
            # Test that the training runs with an augly transformation.
            run_integration_test(config)
