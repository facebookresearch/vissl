# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
from typing import Callable

import torch
import torchvision.transforms as transforms
from PIL import Image
from vissl.data.ssl_transforms.img_pil_color_distortion import ImgPilColorDistortion
from vissl.data.ssl_transforms.img_pil_gaussian_blur import ImgPilGaussianBlur
from vissl.data.ssl_transforms.img_pil_random_photometric import ImgPilRandomPhotometric
from vissl.data.ssl_transforms.img_pil_to_lab_tensor import ImgPil2LabTensor
from vissl.data.ssl_transforms.img_pil_to_multicrop import ImgPilToMultiCrop
from vissl.data.ssl_transforms.img_pil_to_patches_and_image import (
    ImgPilToPatchesAndImage,
)
from vissl.data.ssl_transforms.img_pil_to_tensor import ImgToTensor
from vissl.data.ssl_transforms.img_rotate_pil import ImgRotatePil


"""
Small util to test the throughput of the existing transforms on a cpu host
"""

RAND_TENSOR = (torch.rand((224, 224, 3)) * 255).to(dtype=torch.uint8)
RAND_PIL = Image.fromarray(RAND_TENSOR.numpy())
ITERATIONS = 1000
N_QUEUES = 10  # Simulate the load of N dataloader queues


def benchmark(transform: Callable, title: str, requires_pil: bool = False) -> float:
    """ Given a transform, simulate a real-world load by creating multiple workers """
    results = []

    def store_result(result):
        global results
        results.append(result)

    def worker_load(rank: int):
        print(f"worker {rank}")
        load_one_queue(transform, requires_pil)

    pool = torch.multiprocessing.Pool(N_QUEUES)
    pool.map_async(worker_load, range(N_QUEUES), callback=store_result)
    pool.close()
    pool.join()
    fps = sum(results) / float(N_QUEUES)
    print("{: <20}: {: >10.1f} fps".format(title, fps))


def load_one_queue(transform: Callable, requires_pil: bool = False) -> float:
    """ Run a given transform repeatedly to simulate a single dataloader worker"""
    start = time.time()
    for _ in range(ITERATIONS):
        transform(RAND_PIL) if requires_pil else transform(RAND_TENSOR)

    return ITERATIONS / (time.time() - start)


def testBlur():
    transform = ImgPilGaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0)
    benchmark(transform, "Blur", requires_pil=True)


def testImPil2Lab():
    benchmark(ImgPil2LabTensor, "PIL2Lab")


def testColorDistort():
    transform = ImgPilColorDistortion(strength=0.5)
    benchmark(transform, "Color distortion", requires_pil=True)


def testToTensor():
    transform = transforms.ToTensor()
    benchmark(transform, "ToTensor", requires_pil=True)


def testImgToTensor():
    benchmark(ImgToTensor(), "ImgToTensor", requires_pil=True)


def testImgPatch():
    transform = ImgPilToPatchesAndImage()
    benchmark(transform, "Patches", requires_pil=True)


def testImgRotate():
    transform = ImgRotatePil(num_angles=4, num_rotations_per_img=1)
    benchmark(transform, "ImgRotate", requires_pil=True)


def testImgPilToMulticrop():
    transform = ImgPilToMultiCrop(
        total_num_crops=8,
        num_crops=[2, 6],
        size_crops=[224, 96],
        crop_scales=[[0.14, 1], [0.05, 0.14]],
    )

    benchmark(transform, "MultiCrop", requires_pil=True)


def testImgPilRandomPhotometric():
    transform = ImgPilRandomPhotometric(p=0.5)  # reflect a real world use case
    benchmark(transform, "Photometric", requires_pil=True)


if __name__ == "__main__":
    # Run the transforms and print out an average processing speed
    print("\n")
    print(f"Using {N_QUEUES} {'queues' if N_QUEUES >1 else 'queue'}")
    testBlur()
    testColorDistort()
    testToTensor()
    testImgToTensor()
    testImPil2Lab()
    testImgPatch()
    testImgRotate()
    testImgPilToMulticrop()
    testImgPilRandomPhotometric()
    print("\n")
