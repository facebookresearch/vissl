# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from classy_vision.generic.distributed_util import (
    get_cuda_device_index,
    is_distributed_training_run,
)
from classy_vision.tasks import register_task
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from vissl.config import AttrDict
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.fsdp_utils import fsdp_wrapper, is_valid_fsdp_model
from vissl.utils.misc import is_fairscale_sharded_available


@register_task("self_supervision_fsdp_task")
class SelfSupervisionFSDPTask(SelfSupervisionTask):
    def __init__(self, config: AttrDict):
        super().__init__(config)
        # Ensure pytorch AMP type if mixed precision is on.
        if config["MODEL"]["FSDP_CONFIG"]["mixed_precision"]:
            if not (
                config["MODEL"]["AMP_PARAMS"]["USE_AMP"]
                and config["MODEL"]["AMP_PARAMS"]["AMP_TYPE"] == "pytorch"
            ):
                raise ValueError("FSDP's mixed precision requires pytorch AMP")

    def _init_pytorch_grad_scaler(self):
        assert is_fairscale_sharded_available(), (
            "To use FSDP with PyTorch AMP, ShardedGradScaler() "
            "from fairscale is needed. Please upgrade fairscale"
        )
        from fairscale.optim.grad_scaler import ShardedGradScaler

        self.amp_grad_scaler = ShardedGradScaler()
        logging.info("Setting AMP: using ShardedGradScaler")

    def add_dummy_layer(self):
        """
        Unlike DDP, FSDP works fine even no parameter requires any gradient.
        So we can disable the hack done for DDP.
        """
        pass

    def init_distributed_data_parallel_model(self):
        """
        This method overloads the ClassificationTask class's method from ClassyVision.
        """
        if not is_distributed_training_run():
            return

        assert get_cuda_device_index() > -1, "Distributed training not setup correctly"

        # TODO (Min): We can load checkpoint, but it ends up setting the trunk's _is_root
        # flag to true. We need to set it back to None here.
        # Also, right now, the head's weight is only partially loaded from the checkpoint
        # because we dump the checkpoint after the head if wrapped, but loading it before
        # it is wrapped.
        # For very big models, we need re-work the checkpoint logic because we don't have
        # enough memory to load the entire model on one node. We need to use local_state_dict()
        # API to load checkpoint shards.
        for module in self.base_model.trunk.modules():
            if isinstance(module, FSDP):
                module._is_root = None
        for module in self.base_model.heads.modules():
            if isinstance(module, FSDP):
                module._is_root = None

        # Then, wrap the whole model. We replace the base_model since it is used
        # when checkpoint is taken.
        fsdp_config = self.config["MODEL"]["FSDP_CONFIG"]
        self.base_model = fsdp_wrapper(self.base_model, **fsdp_config)
        self.distributed_model = self.base_model
        assert is_valid_fsdp_model(
            self.distributed_model
        ), "FSDP is not setup correctly"
