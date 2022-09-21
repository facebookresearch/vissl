# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging

import torch
from classy_vision.generic.distributed_util import is_distributed_training_run
from classy_vision.generic.util import copy_model_to_gpu
from classy_vision.hooks import ClassyHook
from classy_vision.losses import build_loss
from classy_vision.meters import build_meter
from classy_vision.optim import build_optimizer, build_optimizer_schedulers
from classy_vision.tasks import ClassificationTask, register_task
from classy_vision.tasks.classification_task import AmpType, BroadcastBuffersMode
from fairscale.nn import FullyShardedDataParallel
from iopath.common.file_io import g_pathmgr
from torch.cuda.amp import GradScaler as TorchGradScaler
from vissl.config import AttrDict
from vissl.data import build_dataloader, build_dataset, print_sampler_config
from vissl.models import build_model, convert_sync_bn
from vissl.optimizers import get_optimizer_param_groups
from vissl.utils.activation_checkpointing import manual_gradient_reduction
from vissl.utils.checkpoint import CheckpointLoader
from vissl.utils.ema_model import ModelEmaV2
from vissl.utils.misc import is_apex_available, is_fairscale_sharded_available


if is_apex_available():
    import apex


@register_task("self_supervision_task")
class SelfSupervisionTask(ClassificationTask):
    """
    A task prepares and holds all the components of a training like optimizer, datasets,
    dataloaders, losses, meters etc. Task also contains the variable like training iteration,
    epoch number etc. that are updated during the training.

    We prepare every single component according to the parameter settings user wants
    and specified in the yaml config file.

    Task also supports 2 additional things:
    1) converts the model BatchNorm layers to the synchronized batchnorm
    2) sets mixed precision (apex and pytorch both supported)
    """

    def __init__(self, config: AttrDict):
        super().__init__()
        self.config = config
        self.checkpoint_path = None

        # Register the task to the proper device (cpu, gpu, ...)
        self.set_device()

        self.checkpoint_folder = None
        self.checkpoint = None
        self.available_splits = []
        self.base_loss = None
        self.meters = None
        self.datasets = None
        self.phases = []
        self.hooks = []
        self.base_model = None
        self.optimizer = None
        self.amp_args = None
        self.amp_type = None
        self.amp_grad_scaler = None
        self.data_and_label_keys = []
        self.set_amp_args()
        self._enable_manual_gradient_reduction = None
        # total number of parameter updates applied to the model by optimizer
        self.num_updates = 0
        # measure time of several training components (data, forward, backward etc..)
        self.perf_stats = None
        # total number of phases including test + train
        self.num_phases = -1  # set by the trainer
        # number of train only phases
        self.num_train_phases = -1  # set by the prepare method
        # number or train epochs set to num_train_phases
        self.num_epochs = -1  # set by the trainer
        # total number of "training" iterations. Inferred from dataloader length and
        # num_train_phases
        self.max_iteration = -1  # set by trainer
        # Current phase id (includes train/test). Starts from 0
        self.phase_idx = -1
        # id of the current training phase training is at. Starts from 0
        self.train_phase_idx = -1  # set by trainer
        # metrics stored during the training.
        self.metrics = {}  # set by the trainer
        self.start_time = -1  # set by trainer
        # time of each batch in training and testing. This can be used to get average
        # batch time etc. batch_time is appended after every parameter update.
        self.batch_time = []  # set by trainer
        # we maintain and store the iteration in the state itself. It counts
        # total number of iterations we do in training phases. Updated
        # after every forward pass of training step.
        # Starts from 1
        self.iteration = 0
        # collect how many total iterations we make irrespective of train/test phase.
        # Useful for debugging purposes. Starts from 1.
        self.local_iteration_num = -1  # set by trainer
        # for every phase, record the start time. Reset at the beginning of each phase
        # by SetDataSamplerEpochHook hook.
        self.phase_start_time = -1  # set by the hook at start of each epoch or phase
        # for every phase, record the number of batches seen. Incremented after every
        # forward pass. Reset at the start of each phase by
        # SetDataSamplerEpochHook hook. Useful for debugging.
        self.batches = -1  # set by the hook at start of each epoch or phase
        # loss curve. Reset at start of each phase/epoch by SetDataSamplerEpochHook hook.
        self.losses = []  # set by the hook at start of each epoch or phase
        # set the bucket_cap_mb for gradient reduction. This can be tuned to overlap
        # communication as much as possible
        self.set_ddp_bucket_cap_mb()
        self.use_gpu = self.device.type == "cuda"
        # optionally save the exponential moving average (ema) of the base_model.
        # and/or run the meters on the ema of the base_model.
        self.ema_model = None
        self.ema_meters = []

    def set_device(self):
        """
        Set the training device: whether gpu or cpu. We use the self.device
        in the rest of the workflow to determine if we should do cpu only training
        or use gpu. set MACHINE.DEVICE = "gpu" or "cpu"
        """
        try:
            self.device = torch.device(
                "cuda" if self.config.MACHINE.DEVICE == "gpu" else "cpu"
            )
        except AttributeError:
            self.device = torch.device("cuda")

    def set_ddp_bucket_cap_mb(self):
        """
        PyTorch DDP supports setting the bucket_cap_mb for all reduce. Tuning
        this parameter can help with the speed of the model. We use the default
        pytorch value of 25MB.
        """
        self.ddp_bucket_cap_mb = self.config.DATA.DDP_BUCKET_CAP_MB
        assert self.ddp_bucket_cap_mb > 0, "bucket_cap_mb must be positive"

    def set_available_splits(self):
        """
        Given the data settings, we determine if we are using both train and test
        datasets. If TEST_MODEL=true, we will add the test to the available_splits.
        If TEST_ONLY=false, we add train to the split as well.
        """
        if self.config.TEST_MODEL:
            self.available_splits.append("TEST")
        if not self.config.TEST_ONLY:
            self.available_splits.append("TRAIN")
        return self

    def set_amp_args(self):
        """
        Two automatic mixed precision implementations are available: Apex's and PyTorch's.

        - If Apex's AMP is enabled, amp_args is a dictionary containing arguments
        to be passed to amp.initialize. Set to None to disable amp.
        To enable mixed precision training, pass amp_args={"opt_level": "O1"} here.
        See https://nvidia.github.io/apex/amp.html for more info.

        - If Pytorch's AMP is enabled, no arguments are needed.
        """

        if self.config.MODEL.AMP_PARAMS.USE_AMP:
            assert (
                self.device.type == "cuda"
            ), "Mixed precision is only available on CUDA devices for now"

            # This will rightly fail if the setting is not correct
            self.amp_type = AmpType[self.config.MODEL.AMP_PARAMS.AMP_TYPE.upper()]
            if self.amp_type == AmpType.APEX:
                self._init_apex_grad_scaler()
            elif self.amp_type == AmpType.PYTORCH:
                self._init_pytorch_grad_scaler()
            logging.info(f"Setting AMP: {self.amp_type} - args: {self.amp_args}")

        else:
            self.amp_args, self.amp_type = None, None
            logging.info("Not using Automatic Mixed Precision")

    def _init_apex_grad_scaler(self):
        # Check Apex availability
        if not is_apex_available():
            raise RuntimeError("Apex is not available. Can't use mixed precision")

        # "amp_args" are actually Apex Amp args
        self.amp_args = self.config.MODEL.AMP_PARAMS.AMP_ARGS
        logging.info(f"Setting AMP: using apex, args {self.amp_args}")

    def _init_pytorch_grad_scaler(self):
        if self.config["OPTIMIZER"]["name"] == "zero":
            assert is_fairscale_sharded_available(), (
                "To use ZeRO with PyTorch AMP, ShardedGradScaler() "
                "from fairscale is needed. Please upgrade fairscale"
            )
            from fairscale.optim.grad_scaler import ShardedGradScaler

            self.amp_grad_scaler = ShardedGradScaler()
            logging.info("Setting AMP: using sharded grad scaler")
        else:
            self.amp_grad_scaler = TorchGradScaler()
            logging.info("Setting AMP: using pytorch grad scaler")

    def set_checkpoint_path(self, checkpoint_path: str):
        """
        Set the checkpoint path for the training
        """
        self.checkpoint_path = checkpoint_path

    def set_checkpoint_folder(self, checkpoint_folder: str):
        """
        Set the checkpoint folder for the training
        """
        self.checkpoint_folder = checkpoint_folder

    def set_iteration(self, iteration):
        """
        Set the iteration number.
        we maintain and store the iteration in the state itself. It counts
        total number of iterations we do in training phases. Updated
        after every forward pass of training step.
        Starts from 1
        """
        assert iteration >= 0, "Iteration number must be positive"
        self.iteration = iteration

    @property
    def enable_manual_gradient_reduction(self) -> bool:
        """
        Lazily initial the enable flag once when model is not None.
        """
        if self._enable_manual_gradient_reduction is None and self.model is not None:
            self.set_manual_gradient_reduction()
        if self._enable_manual_gradient_reduction:
            return True
        return False

    def set_manual_gradient_reduction(self) -> None:
        """
        Called during __init__ to set a flag if manual gradient reduction is enabled.
        """
        assert self.model is not None
        self._enable_manual_gradient_reduction = manual_gradient_reduction(
            self.model, self.config["DISTRIBUTED"]["MANUAL_GRADIENT_REDUCTION"]
        )
        if self._enable_manual_gradient_reduction:
            logging.info("Enabling manual gradient reduction")

    @classmethod
    def from_config(cls, config):
        """
        Create the task from the yaml config input.
        """
        test_only = config.TEST_ONLY

        return (
            cls(config)
            .set_available_splits()
            .set_test_only(test_only)
            .set_epoch_phase_info()
        )

    def set_epoch_phase_info(self):
        # In case optimizer doesn't exist. E.g. for feature extraction.
        optimizer = getattr(self.config, "OPTIMIZER", {})
        self.num_epochs = getattr(optimizer, "num_epochs", 1)
        self.num_train_phases_per_epoch = getattr(
            self.config["DATA"]["TRAIN"], "TRAIN_PHASES_PER_EPOCH", 1
        )
        self.num_train_phases = (
            self.config["OPTIMIZER"]["num_epochs"] * self.num_train_phases_per_epoch
        )

        return self

    # We keep the function because this is used by hooks like checkpoint etc.
    def get_config(self):
        """
        Utility function to store and use the config that was used for the given
        training.
        """
        return {"config": self.config}

    def _build_phases(self):
        """
        Returns list of phases from config. These phases will look like:
        {
          train: is this a train or test phase (bool)?
        }
        If this is a test only run, then only test phases will be
        generated, if this is a training run, then #phases = #train-phases + #test-phases,
        interleaved. We also add the test phases every TEST_EVERY_NUM_EPOCH if
        we don't want the tst to run after every test phase.
        """
        if not self.config["TEST_ONLY"]:
            phases = [{"train": True} for _ in range(self.num_train_phases)]
            # whether the model is train or test only. If the model is not test
            # only, then whether we do test as well or not, is decided from the
            # config file.
            test_every = (
                self.config.get("TEST_EVERY_NUM_EPOCH", 1)
                * self.num_train_phases_per_epoch
            )
            output_phases = []
            for idx, phase in enumerate(phases):
                output_phases.append(phase)

                if idx % test_every == 0 or idx == (len(phases) - 1):
                    output_phases.append({"train": False})
            # we do a little surgery here. Either the phases are test only or
            # [train + test] both interleaved. If we don't want the model to be tested
            # at all (which is sometimes the case in self-supervised learning), we
            # remove the test phases.
            if not self.config["TEST_MODEL"]:
                output_phases = [phase for phase in output_phases if phase["train"]]
        else:
            output_phases = [{"train": False} for _ in range(self.num_train_phases)]
        return output_phases

    def build_datasets(self, current_train_phase_idx=0):
        """
        Get the datasets for the data splits we will use in the training. The
        set_available_splits variable determines the splits used in the training.
        """
        datasets, data_and_label_keys = {}, {}
        for split in self.available_splits:
            datasets[split.lower()] = build_dataset(
                cfg=self.config,
                split=split,
                current_train_phase_idx=current_train_phase_idx,
            )
            data_and_label_keys["input"] = self.config.DATA[split].INPUT_KEY_NAMES
            data_and_label_keys["target"] = self.config.DATA[split].TARGET_KEY_NAMES

        return datasets, data_and_label_keys

    def build_dataloaders(
        self, pin_memory: bool, current_train_phase_idx=0
    ) -> torch.utils.data.DataLoader:
        """
        Build PyTorch dataloaders for all the available_splits. By default, we construct the
        standard PyTorch Dataloader and allow setting all dataloader options.
        """
        # Gives sampler same seed for entire distributed group as per pytorch documentation.
        sampler_seed = self.config["SEED_VALUE"]

        loaders = {
            split.lower(): build_dataloader(
                dataset=self.datasets[split.lower()],
                dataset_config=self.config["DATA"][split],
                num_dataloader_workers=self.config.DATA.NUM_DATALOADER_WORKERS,
                pin_memory=pin_memory,
                multi_processing_method=self.config.MULTI_PROCESSING_METHOD,
                device=self.device,
                sampler_seed=sampler_seed,
                split=split.lower(),
            )
            for split in self.available_splits
        }

        return loaders

    def get_global_batchsize(self):
        """
        Return global batchsize used in the training across all the trainers.
        We check what phase we  are in (train or test) and get the dataset
        used in that phase. We call get_global_batchsize() of the dataset.
        """
        for phase_type in self.datasets:
            if phase_type.lower() == self.phase_type.lower():
                return self.datasets[phase_type].get_global_batchsize()
        raise ValueError(f"{self.phase_type} not found in self.datasets")

    def _build_optimizer(self):
        """
        Build optimizers using the optimizer settings specified by user.
        For SGD, we support LARC as well. In order to use LARC, Apex must
        be installed.
        """
        optimizer_config = self.config["OPTIMIZER"]
        if optimizer_config.use_larc and optimizer_config.name != "sgd_fsdp":
            assert is_apex_available(), "Apex must be available to use LARC"
        optim = build_optimizer(optimizer_config)
        return optim

    def _build_optimizer_schedulers(self):
        """
        Build the param schedulers to be used in training.
        """
        return build_optimizer_schedulers(self.config["OPTIMIZER"])

    def _build_loss(self):
        """
        Build the loss used in training. Supports all PyTorch losses
        and custom defined losses.

        For some losses that require memory banks (for example in info_nce loss),
        we need to store the size of data as we use it to allocate memory.
        Since dataset size is not known at the time of config parsing, we set
        the data size parameter here.
        """
        # in some cases like memory bank, we need to store the size of data
        # as we use it to allocate memory. Hence we set that parameter here.
        logging.info("Building loss...")
        loss_name = self.config.LOSS["name"]
        assert loss_name in list(self.config.LOSS.keys()), (
            f"Loss {loss_name} params unknown. The loss name and the param dict "
            f"key name should match. Known: {list(self.config.LOSS.keys())}"
        )
        loss_config = self.config.LOSS[loss_name]
        if "num_train_samples" in loss_config.keys():
            for split in self.available_splits:
                if split == "TRAIN":
                    loss_config["num_train_samples"] = len(self.datasets["train"])
                if split == "TEST":
                    loss_config["num_train_samples"] = len(self.datasets["test"])
        loss_config["name"] = loss_name
        loss = build_loss(loss_config)
        return loss

    def _build_meters(self):
        """
        Returns meters for task.
        """
        meter_names = self.config["METERS"].get("names", [])

        if not meter_names:
            return []

        meters = []
        for meter_name in meter_names:
            meter_params = self.config["METERS"][meter_name]
            meter_config = {"name": meter_name, **meter_params}
            meters.append(build_meter(meter_config))

        return meters

    def _restore_model_weights(self, model, strict: bool = False):
        """
        If using a weights file to initialize the model, we load the weights
        and initialize the model. Since the weights file specified
        by user might not be VISSL trained weights, we expose several config
        options like APPEND_PREFIX, etc to allow successful loading of the weights.
        See MODEL.WEIGHTS_INIT description in vissl/config/defaults.yaml for details.
        """
        params_from_file = self.config["MODEL"]["WEIGHTS_INIT"]
        init_weights_path = params_from_file["PARAMS_FILE"]
        assert init_weights_path, "Shouldn't call this when init_weight_path is empty"
        logging.info(f"Initializing model from: {init_weights_path}")

        if g_pathmgr.exists(init_weights_path):
            checkpoint = CheckpointLoader.load_and_broadcast_init_weights(
                checkpoint_path=init_weights_path, device=torch.device("cpu")
            )
            logging.info(f"Checkpoint loaded: {init_weights_path}...")
            model.init_model_from_weights_params_file(
                self.config, checkpoint, strict=strict
            )
        return model

    def _build_model(self, strict_load: bool = False):
        """
        - Builds and returns model used for task. The returned model is not copied to
          gpu yet (if using gpu) and neither wrapped with DDP yet. This is done later
          by self.prepare()

        - We also convert the model BatchNorm layers to SyncBatchNorm if user
          has set the config option. We support PyTorch and Apex SyncBatchNorms
          both.

        - If the model is set to be in evaluation model and the full model must be frozen,
          we freeze the model.

        - If the model must be initialized from a checkpoint or user passed weights file
          we initialize the model from the checkpoint or the weights.
        """
        logging.info("Building model....")

        # Instantiate the raw model as specified
        model = build_model(self.config["MODEL"], self.config["OPTIMIZER"])

        # Convert the BatchNorm layers to SyncBatchNorm if needed
        # Both Apex and Pytorch SyncBatchNorms are GPU only
        if (
            self.config["MODEL"]["SYNC_BN_CONFIG"]["CONVERT_BN_TO_SYNC_BN"]
            and self.config["MACHINE"]["DEVICE"] == "gpu"
        ):
            model = convert_sync_bn(self.config, model)

        # Enforce eval mode, no matter what the prior tranforms have done.
        # For instance apex converts batch-norms and sets `requires_grad` to True
        if self.config["MODEL"]["FEATURE_EVAL_SETTINGS"]["EVAL_MODE_ON"]:
            if self.config["MODEL"]["FEATURE_EVAL_SETTINGS"]["FREEZE_TRUNK_ONLY"]:
                logging.info(
                    "config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True, "
                    "will freeze trunk..."
                )
                model.freeze_trunk()
            elif self.config["MODEL"]["FEATURE_EVAL_SETTINGS"]["FREEZE_TRUNK_AND_HEAD"]:
                logging.info(
                    "config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD=True, will "
                    "freeze trunk and head..."
                )
                model.freeze_head_and_trunk()

        # assert that if the user set the PARAMS_FILE, it must exist and be valid.
        if (
            self.checkpoint_path is None
            and self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
        ):
            assert g_pathmgr.exists(
                self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
            ), "Specified PARAMS_FILE does NOT exist"
        # If we want to initialize the model in case of finetuning or evaluation,
        # we do it here. But we check that there is no checkpoint existing before
        # This is important in cases when the model training dies.
        if (
            self.checkpoint_path is None
            and self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
            and g_pathmgr.exists(self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"])
        ):
            model = self._restore_model_weights(model, strict=strict_load)

        return model

    def init_distributed_data_parallel_model(self):
        """
        This method overloads the ClassificationTask class's method from ClassyVision.
        """
        if not is_distributed_training_run():
            return

        for module in self.base_model.modules():
            if isinstance(module, FullyShardedDataParallel):
                raise ValueError(
                    "DistributedDataParallel should not be used"
                    "with a FullyShardedDataParallel model.\n"
                    "Please set config.TRAINER.TASK_NAME='self_supervision_fsdp_task'"
                )

        # Make sure that DistributedDataParallel will be happy
        if not any((p.requires_grad for p in module.parameters())):
            self.add_dummy_layer()

        super().init_distributed_data_parallel_model()

    def set_epoch(
        self, phase_type: str, epoch: int, start_iter: int, train_phase_idx: int
    ):
        if hasattr(self.dataloaders[phase_type], "sampler"):
            sampler = self.dataloaders[phase_type].sampler
            # (Re-)Shuffle data: set epoch of distributed (or fairstore) sampler
            # Resume from the iteration if valid
            self.set_train_epoch_start_iter(sampler, epoch, start_iter, train_phase_idx)
            print_sampler_config(sampler)

        # call set_epoch and set_start_iter for AirstoreDataset since it handles
        # shuffle and sample skipping behavior internally
        dataset = self.datasets[phase_type]
        if hasattr(dataset, "data_objs"):
            for data_obj in dataset.data_objs:
                self.set_train_epoch_start_iter(
                    data_obj, epoch, start_iter, train_phase_idx
                )

    def set_train_epoch_start_iter(
        self, dataset_or_sampler, epoch: int, start_iter: int, train_phase_idx: int
    ):
        # (Re-)Shuffle data: set epoch of distributed (or fairstore) sampler
        if hasattr(dataset_or_sampler, "set_epoch"):
            dataset_or_sampler.set_epoch(epoch)
        # Resume from the iteration if valid
        if hasattr(dataset_or_sampler, "set_start_iter"):
            dataset_or_sampler.set_start_iter(start_iter)

        if hasattr(dataset_or_sampler, "set_train_phase_idx"):
            dataset_or_sampler.set_train_phase_idx(train_phase_idx)

    def num_phase_samples(self, phase_type: str) -> int:
        """
        Number of samples in a phase.
        """
        dataset = self.datasets[phase_type.lower()]
        return dataset.num_samples()

    def _compute_start_iter_from_checkpoint(self, phase_type) -> int:
        # used for calculating the start iteration (count from current epoch) when resuming
        # from checkpoint
        if self.checkpoint is None or self.checkpoint["iteration"] <= 0:
            return 0

        num_iters_in_epochs = len(self.dataloaders[phase_type])
        num_epochs = self.checkpoint["train_phase_idx"] + 1
        num_train_iters_done = num_epochs * num_iters_in_epochs
        return self.checkpoint["iteration"] - num_train_iters_done

    def recreate_data_iterator(
        self,
        phase_type: str,
        epoch: int,
        compute_start_iter: bool,
        train_phase_idx: int,
    ):
        """
        Recreate data iterator (including multiprocessing workers) and destroy the
        previous iterators.

        This is called when we load a new checkpoint or when phase changes during
        the training (one epoch to the next).
        DataSampler may need to be informed on those events to update the
        epoch and start_iteration so that the data is deterministically shuffled,
        so we call them here.
        """
        start_iter = 0
        if compute_start_iter:
            start_iter = self._compute_start_iter_from_checkpoint(phase_type)

        self.set_epoch(phase_type, epoch, start_iter, train_phase_idx)

        # Gives sampler same seed for entire distributed group as per pytorch documentation.
        sampler_seed = self.config["SEED_VALUE"]
        dataset = self.datasets[phase_type]

        # For OSS, this will always return false.
        # Otherwise, we will rebuild the dataloader after every phase.
        if dataset.rebuild_dataloader():
            dataloader = build_dataloader(
                dataset=dataset,
                dataset_config=self.config.DATA[phase_type.upper()],
                num_dataloader_workers=self.config.DATA.NUM_DATALOADER_WORKERS,
                pin_memory=self.config.DATA.PIN_MEMORY,
                multi_processing_method=self.config.MULTI_PROCESSING_METHOD,
                device=self.device,
                sampler_seed=sampler_seed,
                split=phase_type,
            )

            # delete old dataloader and reset it.
            del self.dataloaders[phase_type]
            gc.collect()
            self.dataloaders[phase_type] = dataloader

        # delete old dataiterator and reset it.
        del self.data_iterator
        gc.collect()
        self.data_iterator = iter(self.dataloaders[phase_type])

    def _set_classy_state(self, state):
        """
        We load/set the model state setting here to resume correctly from the
        specified state. Usually called when resuming training from a previous
        model checkpoint.
        We set the model phase (train or eval), model weights,
        copy the model to correct device, initialize meters, initialize optimizers
        initialize amp state, set loss state, set the train phase number, iteration,
        recreate data iterators, etc.
        """
        logging.info("=======Updating classy state_dict from checkpoint=======")
        # here we load the state specific things only. The other extra variables
        # are init from the checkpoint in the trainer step.
        self.train = state["train"]
        self.base_model.set_classy_state(state["base_model"])
        # We need to set the model on correct device here unlike in the case of
        # training from scratch. The optimizer looks at the model parameters like
        # momentum etc. for getting the device info. Since in case of scratch
        # training, we don't have those and the optimizer just gets the inputs
        # as cuda inputs from the model, it can work. However, when we load from
        # a checkpoint, we already have these parameters and the type is CPU
        # (since the model isn't copied to gpu yet). The copy_model_to_gpu()
        # doesn't modify optimizer params device. The optimizer is constructed
        # with the CPU inputs. When the model runs, it rather sends CUDA.
        self.base_model.to(self.device)

        self._set_ema_model_state(state)

        for meter, meter_state in zip(self.meters, state["meters"]):
            meter.set_classy_state(meter_state)
        self.optimizer.set_classy_state(state["optimizer"])

        # restore amp state. It's called after amp.initialize is done.
        if "amp" in state:
            if self.amp_type == AmpType.APEX:
                if is_apex_available():
                    apex.amp.load_state_dict(state["amp"])
                else:
                    logging.warning(
                        "Loading a checkpoint which has amp state but apex isn't available now"
                    )
            else:
                self.amp_grad_scaler.load_state_dict(state["amp"])
        self.phase_idx = state["phase_idx"]
        self.train_phase_idx = state["train_phase_idx"]
        self.num_updates = state["num_updates"]
        self.losses = state["losses"]

        phase_type = "train" if self.train else "test"
        phase = self.phases[self.phase_idx]

        # Re-create the data iterator.
        # We are restoring from a checkpoint, which means we need to
        #   (1) set the right epoch
        #   (2) set the right start_iter
        # epoch number is `phase_idx + 1` since checkpoint's value is the epoch finished.
        # start_iter is computed in recreate_data_iterator based on iteration
        # number from the checkpoint state.
        self.recreate_data_iterator(
            phase_type,
            epoch=self.phase_idx + 1,
            compute_start_iter=True,
            train_phase_idx=self.train_phase_idx + 1,
        )

        # set the model to train or eval depending on what phase we are in
        self.base_model.train(phase["train"])

        if self.train and self.train_phase_idx >= 0:
            self.optimizer.on_epoch(self.where)

    def _set_ema_model_state(self, state):
        """
        Only used if EmaMetersHook is enabled.
        """
        if self.ema_model is not None:
            logging.info("Loading ema model")
            self.ema_model.module.set_classy_state(state["ema_model"])
            for meter, meter_state in zip(self.ema_meters, state["ema_meters"]):
                meter.set_classy_state(meter_state)

    def _update_classy_state(self, state_dict=None):
        """
        Updates classy state with the provided state dict from a checkpoint.
        state_dict = checkpoint loaded state
        """
        if state_dict is not None:
            try:
                self._set_classy_state(state_dict)
                success = True
            except Exception as e:
                logging.exception(f"Could not load the checkpoint: {e}")
                success = False
            assert success, "Update classy state from checkpoint failed."
        return self

    def _set_ddp_options(self):
        """
        set DDP options if the user has supplied them
        """
        broadcast_buffers = self.config["DISTRIBUTED"]["BROADCAST_BUFFERS"]
        if broadcast_buffers:
            logging.info(
                "Broadcast model BN buffers from primary on every forward pass"
            )
            broadcast_buffers_enum_mode = BroadcastBuffersMode.FORWARD_PASS
            self.set_distributed_options(
                broadcast_buffers_mode=broadcast_buffers_enum_mode
            )  # NOQA

    def run_hooks(self, hook_function_name: str):
        """
        Override the ClassyTask run_hook function and run the hooks whenever called
        """
        for hook in self.hooks:
            getattr(hook, hook_function_name, ClassyHook._noop)(self)

    def prepare_optimizer(self):
        """
        Constructs the optimizer using the user defined settings in the yaml config.
        The model must be on the correct device (cuda or cpu) by this point.
        """
        param_groups = get_optimizer_param_groups(
            model=self.base_model,
            model_config=self.config["MODEL"],
            optimizer_config=self.config["OPTIMIZER"],
            optimizer_schedulers=self.optimizer_schedulers,
        )
        self.optimizer.set_param_groups(param_groups)

    def prepare(self, pin_memory: bool = False):
        """
        Prepares the task:
        - dataloaders
        - model
        - copy model to correct device
        - meters
        - loss
        - optimizer
        - LR schedulers
        - AMP state
        - resume from a checkpoint if available
        """
        self.phases = self._build_phases()
        self.num_phases = len(self.phases)
        self.base_model = self._build_model()
        self._set_ddp_options()
        self.meters = self._build_meters()
        self.optimizer = self._build_optimizer()
        self.optimizer_schedulers = self._build_optimizer_schedulers()

        if self.device.type == "cuda":
            self.base_model = copy_model_to_gpu(self.base_model)

        # initialize the pytorch optimizer now since the model has been moved to
        # the appropriate device.
        self.prepare_optimizer()

        # Enable mixed precision grad scalers
        if self.amp_type == AmpType.APEX:
            # Allow Apex Amp to perform casts as specified by the amp_args.
            # This updates the model and the PyTorch optimizer (which is wrapped
            # by the ClassyOptimizer in self.optimizer).
            # NOTE: this must happen before loading the checkpoint. See
            # https://nvidia.github.io/apex/amp.html#checkpointing for more details.
            self.base_model, self.optimizer.optimizer = apex.amp.initialize(
                self.base_model, self.optimizer.optimizer, **self.amp_args
            )

        # Create EMA average of the model if hook is specified.
        ema_config = self.config["HOOKS"]["EMA_MODEL"]
        if ema_config["ENABLE_EMA_METERS"] or ema_config["SAVE_EMA_MODEL"]:
            self._create_ema_model()

        # Restore an hypothetical checkpoint
        # - For DDP model, the load will load the full model on all ranks
        # - For FSDP model, the load will automatically dispatch to the shard
        #   to be loaded by the current rank
        vissl_state_dict = None
        if self.checkpoint_path is not None:
            self.checkpoint = CheckpointLoader.load_and_broadcast_checkpoint(
                checkpoint_folder=self.checkpoint_folder,
                checkpoint_path=self.checkpoint_path,
                device=torch.device("cpu"),
            )
            if self.checkpoint is not None:
                self.iteration = self.checkpoint["iteration"]
                self.local_iteration_num = self.checkpoint["iteration_num"]
                vissl_state_dict = self.checkpoint.get("classy_state_dict")
            else:
                raise ValueError(f"Could not load checkpoint: {self.checkpoint_path}")

        current_train_phase_idx = (
            vissl_state_dict["train_phase_idx"] + 1 if vissl_state_dict else 0
        )

        self.datasets, self.data_and_label_keys = self.build_datasets(
            current_train_phase_idx
        )

        # set dataset state before building dataloader, in order to capture checkpoint info.
        if vissl_state_dict and "train" in self.datasets:
            self.datasets["train"].set_classy_state(
                vissl_state_dict.get("train_dataset_iterator")
            )

        self.dataloaders = self.build_dataloaders(
            pin_memory=pin_memory, current_train_phase_idx=current_train_phase_idx
        )

        # Build base loss, move to device, and load from checkpoint if applicable
        self.base_loss = self._build_loss()
        self.base_loss = self.base_loss.to(self.device)
        if self.checkpoint and "loss" in self.checkpoint:
            self.base_loss.load_state_dict(self.checkpoint["loss"])
            logging.info("======Loaded loss state from checkpoint======")

        return self._update_classy_state(vissl_state_dict)

    def prepare_extraction(self, pin_memory: bool = False):
        """
        Prepares a light-weight task for feature extraction on multi-gpu. The model
        runs in eval mode only.
        """
        self.datasets, self.data_and_label_keys = self.build_datasets()
        self.dataloaders = self.build_dataloaders(pin_memory=pin_memory)
        # build the meters in case the extraction is for predictions.
        self.meters = self._build_meters()
        self.base_model = self._build_model(strict_load=True)
        if self.device.type == "cuda":
            self.base_model = copy_model_to_gpu(self.base_model)
        return self

    def add_dummy_layer(self):
        """
        In case of feature evaluation mode, if we are freezing both trunk and
        head, DDP won't work as there are no parameters in the model. Adding
        the dummy head will lead to features being not right. So we rather
        add the dummy layer to the model and use DDP. We copy the model to
        gpu (if using gpus) after the new dummy layer addition.
        """
        fully_frozen_model = self.base_model.is_fully_frozen_model()
        if fully_frozen_model:
            self.base_model.dummy_layer = torch.nn.Linear(4, 4)
            if self.device.type == "cuda":
                self.base_model = copy_model_to_gpu(self.base_model)

    def _create_ema_model(self):
        logging.info("Building the EMA model.")
        ema_model = build_model(self.config["MODEL"], self.config["OPTIMIZER"])
        self.ema_model = ModelEmaV2(
            ema_model,
            decay=self.config["HOOKS"]["EMA_MODEL"]["DECAY"],
            device=self.config["HOOKS"]["EMA_MODEL"]["EMA_DEVICE"],
        )
        self.ema_model.set(self.base_model)
