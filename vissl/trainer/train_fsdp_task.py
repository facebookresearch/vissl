# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from classy_vision.generic.distributed_util import (
    get_cuda_device_index,
    is_distributed_training_run,
    is_primary,
)
from classy_vision.tasks import register_task
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.nn import Linear
from vissl.config import AttrDict
from vissl.models.heads.swav_prototypes_head import SwAVPrototypesHead
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.misc import is_fairscale_sharded_available, set_torch_seed


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
        logging.info("Setting AMP: using sharded grad scaler")

    def init_distributed_data_parallel_model(self):
        """
        Initialize FSDP if needed.

        This method overloads the ClassificationTask class's method from ClassyVision.
        """
        if not is_distributed_training_run():
            return

        # Make sure default cuda device is set. TODO (Min): we should ensure FSDP can
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
        if len(self.base_model.heads) != 1:
            raise ValueError(
                f"FSDP only support 1 head, not {len(self.base_model.heads)} heads"
            )
        head0 = self.base_model.heads[0]
        if isinstance(head0, SwAVPrototypesHead):
            # This is important for convergence!
            #
            # Since we "normalize" this layer in the update hook, we need to keep its
            # weights in full precision. It is output is going into the loss and used
            # for clustering, so we need to have that in full precision as well.
            fp_fsdp_config = fsdp_config.copy()
            fp_fsdp_config["flatten_parameters"] = False
            fp_fsdp_config["mixed_precision"] = False
            fp_fsdp_config["fp32_reduce_scatter"] = False
            for j in range(head0.nmb_heads):
                module = getattr(head0, "prototypes" + str(j))
                module = FSDP(module=module, **fp_fsdp_config)
                setattr(head0, "prototypes" + str(j), module)
        head0 = FSDP(module=head0, **fsdp_config)
        self.base_model.heads[0] = head0

        # Init the head properly since the weights are potentially initialized on different
        # ranks with different seeds. We first summon the full params from all workers.
        # Then, within that context, we set a fixed random seed so that all workers init the
        # weights the same way. Finally, we reset the layer's weights using reset_parameters().
        #
        # TODO (Min): This will go away once we have a way to sync from rank 0.
        with head0.summon_full_params():
            with set_torch_seed(self.config["SEED_VALUE"]):
                for m in head0.modules():
                    if isinstance(m, Linear):
                        m.reset_parameters()
        head0._reset_lazy_init()
        head0.prototypes0._reset_lazy_init()

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
