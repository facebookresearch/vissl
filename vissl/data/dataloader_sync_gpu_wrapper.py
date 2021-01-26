# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Iterator

import torch
from classy_vision.dataset.dataloader_wrapper import DataloaderWrapper
from classy_vision.generic.util import recursive_copy_to_gpu


class DataloaderSyncGPUWrapper(DataloaderWrapper):
    """
    Dataloader which wraps another dataloader, and moves the data to GPU
    in async manner so as to overlap the cost of copying data from
    cpu to gpu with the previous model iteration.
    """

    def __init__(self, dataloader: Iterable) -> None:
        assert torch.cuda.is_available(), "This Dataloader wrapper needs a CUDA setup"
        super().__init__(dataloader)
        self._iter = None

    def __iter__(self) -> Iterator[Any]:
        # The wrapped dataloader may have been changed in place
        # rebuild a new iterator and prefetch
        self._iter = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        # Get data from the iterator and move to GPU
        # This can raise `StopIteration`
        return recursive_copy_to_gpu(next(self._iter), non_blocking=True)

    def __len__(self) -> int:
        return len(self.dataloader)
