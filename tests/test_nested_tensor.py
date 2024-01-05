# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from vissl.data.collators.collator_helper import MultiDimensionalTensor


logger = logging.getLogger("__name__")


class TestMultiDimensionalTensor(unittest.TestCase):
    def test_run(self) -> None:
        """
        Test the nested tensor works
        """
        tensor1 = torch.randn(1, 3, 7, 7)
        tensor2 = torch.randn(1, 3, 4, 4)
        out = MultiDimensionalTensor.from_tensors([tensor1, tensor2])
        padded_tensor, out_mask = out.tensor, out.mask

        # check the output shapes are good
        self.assertEqual(padded_tensor.shape, (2, 3, 7, 7), padded_tensor.shape)
        self.assertEqual(out_mask.shape, (2, 7, 7), out_mask.shape)

        # check that the output mask is as expected. The padded
        # indexes should have 1.0 value in the mask
        self.assertTrue(
            out_mask.float().equal(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        ],
                    ]
                )
            )
        )
