# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from classy_vision.optim.zero import ZeRO
from classy_vision.tasks import register_task
from classy_vision.tasks.classification_task import BroadcastBuffersMode
from fairscale.nn.data_parallel import ShardedDataParallel
from vissl.config import AttrDict
from vissl.trainer.train_task import SelfSupervisionTask


# More information on ShardedDDP can be found in the Fairscale repository
# https://github.com/facebookresearch/fairscale


@register_task("self_supervision_sdp_task")
class SelfSupervisionSDPTask(SelfSupervisionTask):
    def __init__(self, config: AttrDict):
        super().__init__(config)

    def init_distributed_data_parallel_model(self):
        """
        Initialize ShardedDataParallel, needed for sharded distributed training.
        This is where a model should be wrapped by DDP.
        """
        broadcast_buffers = (
            self.broadcast_buffers_mode == BroadcastBuffersMode.FORWARD_PASS
        )

        # Replace the original DDP wrap by the shard-aware ShardedDDP
        # we use the fairscale reduce_buffer_size by default however, if user sets it to
        # some different value, we use the different value.
        reduce_buffer_size = 2**23
        if self.config.MODEL.SHARDED_DDP_SETUP.reduce_buffer_size >= 0:
            reduce_buffer_size = self.config.MODEL.SHARDED_DDP_SETUP.reduce_buffer_size
        logging.info(f"Setting reduce_buffer_size: {reduce_buffer_size}")
        if isinstance(self.optimizer, ZeRO):
            logging.info("Using ShardedDDP")
            self.distributed_model = ShardedDataParallel(
                module=self.base_model,
                sharded_optimizer=self.optimizer.optimizer,
                broadcast_buffers=broadcast_buffers,
                reduce_buffer_size=reduce_buffer_size,
            )
        else:
            raise NotImplementedError(
                "This DataParallel engine should only be used in conjunction with ZeRO"
            )
