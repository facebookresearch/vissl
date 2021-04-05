# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from classy_vision.generic.distributed_util import (
    get_cuda_device_index,
    is_distributed_training_run,
)
from classy_vision.generic.distributed_util import is_primary
from classy_vision.tasks import register_task
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from vissl.config import AttrDict
from vissl.models.heads.swav_prototypes_head import SwAVPrototypesHead
from vissl.trainer.train_task import SelfSupervisionTask


@register_task("self_supervision_fsdp_task")
class SelfSupervisionFSDPTask(SelfSupervisionTask):
    def __init__(self, config: AttrDict):
        super().__init__(config)
        # assert AMP is off since FSDP has its own mixed precision.
        assert not config["MODEL"]["AMP_PARAMS"]["USE_AMP"], (
            "FSDP has its own mixed precision. We turn off Apex/Torch AMP to avoid "
            "and conflict and extra GPU memory usage."
        )

    def init_distributed_data_parallel_model(self):
        """
        Initialize FSDP if needed.

        This method overloads the ClassificationTask class's method from ClassyVision.
        """
        if not is_distributed_training_run():
            return

        # Make sure default cuda device is set. TODO (Min): we should enable FSDP can
        # be enabled for 1-GPU as well, but the use case there is likely different.
        # I.e. perhaps we use it for cpu_offloading.
        assert get_cuda_device_index() > -1, "Distributed training not setup correctly"

        # The model might be already wrapped by FSDP internally. Check regnet_fsdp.py.
        # Here, we wrap it at the outer most level.
        fsdp_config = self.config["MODEL"]["FSDP_CONFIG"]
        if is_primary():
            logging.info(f"Using FSDP, config: {fsdp_config}")

        # First, wrap the head's prototype_i layers if it is SWAV.
        # TODO (Min): make this more general for different models, which may have multiple
        #             heads.
        head0 = self.base_model.heads[0]
        if isinstance(head0, SwAVPrototypesHead):
            for j in range(head0.nmb_heads):
                module = getattr(head0, "prototypes" + str(j))
                module = FSDP(module=module, **fsdp_config)
                setattr(head0, "prototypes" + str(j), module)

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

        # Then, wrap the whole model. We replace the base_model since it is used
        # when checkpoint is taken.
        self.base_model = FSDP(module=self.base_model, **fsdp_config)
        self.distributed_model = self.base_model
