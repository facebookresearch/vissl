# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import gc
import logging

import torch
from classy_vision.generic.util import (
    copy_model_to_gpu,
    load_and_broadcast_checkpoint,
)
from classy_vision.losses import build_loss
from classy_vision.meters import build_meter
from classy_vision.optim import build_optimizer, build_optimizer_schedulers
from classy_vision.tasks import ClassificationTask, register_task
from classy_vision.tasks.classification_task import BroadcastBuffersMode, AmpType
from fairscale.optim.grad_scaler import ShardedGradScaler
from fvcore.common.file_io import PathManager
from torch.cuda.amp import GradScaler as TorchGradScaler
from vissl.data import build_dataset, get_loader, print_sampler_config
from vissl.models import build_model, convert_sync_bn
from vissl.optimizers import get_optimizer_param_groups
from vissl.utils.activation_checkpointing import manual_gradient_reduction
from vissl.utils.checkpoint import init_model_from_weights
from vissl.utils.hydra_config import AttrDict
from vissl.utils.misc import is_apex_available

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
        # batch time etc. batch_time is appended after every parameter update by
        # UpdateTrainBatchTimeHook and if test phase, by UpdateTestBatchTimeHook
        self.batch_time = []  # set by trainer
        # we maintain and store the iteration in the state itself. It counts
        # total number of iterations we do in training phases. Updated
        # after every forward pass of training step in UpdateTrainIterationNumHook.
        # Starts from 1
        self.iteration = 0
        # collect how many total iterations we make irrespective of train/test phase.
        # Useful for debugging purposes. Starts from 1.
        self.local_iteration_num = -1  # set by trainer
        # for every phase, record the start time. Reset at the beginning of each phase
        # by SetDataSamplerEpochHook hook.
        self.phase_start_time = -1  # set by the hook at start of each epoch or phase
        # for every phase, record the number of batches seen. Incremented after every
        # forward pass by UpdateBatchesSeenHook. Reset at the start of each phase by
        # SetDataSamplerEpochHook hook. Useful for debugging.
        self.batches = -1  # set by the hook at start of each epoch or phase
        # loss curve. Reset at start of each phase/epoch by SetDataSamplerEpochHook hook.
        self.losses = []  # set by the hook at start of each epoch or phase
        # set the bucket_cap_mb for gradient reduction. This can be tuned to overlap
        # communication as much as possible
        self.set_ddp_bucket_cap_mb()
        self.use_gpu = self.device.type == "cuda"

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

            # Check Apex availability
            if self.amp_type == AmpType.APEX:
                if not is_apex_available():
                    raise RuntimeError(
                        "Apex is not available. Can't use mixed precision"
                    )

                # "amp_args" are actually Apex Amp args
                self.amp_args = self.config.MODEL.AMP_PARAMS.AMP_ARGS

            elif self.amp_type == AmpType.PYTORCH:
                # If the optimizer is sharded, then the GradScaler needs to be shard-aware
                self.amp_grad_scaler = (
                    ShardedGradScaler()
                    if self.config["OPTIMIZER"]["name"] == "zero"
                    else TorchGradScaler()
                )
            logging.info(f"Setting AMP: {self.amp_type} - args: {self.amp_args}")

        else:
            self.amp_args, self.amp_type = None, None
            logging.info("Not using Automatic Mixed Precision")

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
        after every forward pass of training step in UpdateTrainIterationNumHook.
        Starts from 1
        """
        assert iteration >= 0, "Iteration number must be positive"
        self.iteration = iteration

    @classmethod
    def from_config(cls, config):
        """
        Create the task from the yaml config input.
        """
        test_only = config.TEST_ONLY
        return cls(config).set_available_splits().set_test_only(test_only)

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
        num_epochs = self.config["OPTIMIZER"]["num_epochs"]
        if not self.config["TEST_ONLY"]:
            phases = [{"train": True} for _ in range(num_epochs)]
            # whether the model is train or test only. If the model is not test
            # only, then whether we do test as well or not, is decided from the
            # config file.
            test_every = self.config.get("TEST_EVERY_NUM_EPOCH", 1)
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
            output_phases = [{"train": False} for _ in range(num_epochs)]
        return output_phases

    def build_datasets(self):
        """
        Get the datasets for the data splits we will use in the training. The
        set_available_splits variable determines the splits used in the training.
        """
        datasets, data_and_label_keys = {}, {}
        for split in self.available_splits:
            datasets[split] = build_dataset(self.config, split)
            data_and_label_keys["input"] = self.config.DATA[split].INPUT_KEY_NAMES
            data_and_label_keys["target"] = self.config.DATA[split].TARGET_KEY_NAMES
        return datasets, data_and_label_keys

    def build_dataloaders(self, pin_memory: bool) -> torch.utils.data.DataLoader:
        """
        Build PyTorch dataloaders for all the available_splits. We construct the
        standard PyTorch Dataloader and allow setting all dataloader options.
        """
        self.datasets, self.data_and_label_keys = self.build_datasets()

        loaders = {
            split.lower(): get_loader(
                dataset=self.datasets[split],
                dataset_config=self.config["DATA"][split],
                num_dataloader_workers=self.config.DATA.NUM_DATALOADER_WORKERS,
                pin_memory=pin_memory,
                multi_processing_method=self.config.MULTI_PROCESSING_METHOD,
                device=self.device,
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
        if optimizer_config.use_larc:
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
                    loss_config["num_train_samples"] = len(self.datasets["TRAIN"])
                if split == "TEST":
                    loss_config["num_train_samples"] = len(self.datasets["TEST"])
        loss_config["name"] = loss_name
        loss = build_loss(loss_config)
        return loss

    def _build_meters(self):
        """
        Returns meters for task.
        """
        meter_name = self.config["METERS"].get("name", "")
        if not meter_name:
            return []
        meter_params = self.config["METERS"][meter_name]
        meter_config = {"name": meter_name, **meter_params}
        return [build_meter(meter_config)]

    def _restore_model_weights(self, model):
        """
        If using a weights file to initialize the model, we load the weights
        and initialize the model. Since the weights file specified
        by user might not be VISSL trained weights, we expose several config
        options like APPEND_PREFIX, etc to allow successful loading of the weights.
        See MODEL.WEIGHTS_INIT description in vissl/config/defaults.yaml for details.
        """
        params_from_file = self.config["MODEL"]["WEIGHTS_INIT"]
        init_weights_path = params_from_file["PARAMS_FILE"]
        logging.info(f"Initializing model from: {init_weights_path}")

        if PathManager.exists(init_weights_path):
            weights = load_and_broadcast_checkpoint(
                init_weights_path, device=torch.device("cpu")
            )
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            replace_prefix = params_from_file.get("REMOVE_PREFIX", None)
            append_prefix = params_from_file.get("APPEND_PREFIX", None)
            state_dict_key_name = params_from_file.get("STATE_DICT_KEY_NAME", None)

            # we initialize the weights from this checkpoint. However, we
            # don't care about the other metadata like iteration number etc.
            # So the method only reads the state_dict
            init_model_from_weights(
                self.config,
                model,
                weights,
                state_dict_key_name=state_dict_key_name,
                skip_layers=skip_layers,
                replace_prefix=replace_prefix,
                append_prefix=append_prefix,
            )
        return model

    def _build_model(self):
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
            assert PathManager.exists(
                self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
            ), "Specified PARAMS_FILE does NOT exist"
        # If we want to initialize the model in case of finetuning or evaluation,
        # we do it here. But we check that there is no checkpoint existing before
        # This is important in cases when the model training dies.
        if self.checkpoint_path is None and PathManager.exists(
            self.config["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
        ):
            model = self._restore_model_weights(model)

        return model

    def recreate_data_iterator(self, phase_type, epoch, compute_start_iter):
        """
        Recreate data iterator (including multiprocessing workers) and destroy the
        previous iterators.

        This is called when we load a new checkpoint or when phase changes during
        the training (one epoch to the next).
        DataSampler may need to be informed on those events to update the
        epoch and start_iteration so that the data is deterministically shuffled,
        so we call them here.
        """
        if hasattr(self.dataloaders[phase_type], "sampler"):
            sampler = self.dataloaders[phase_type].sampler
            # (Re-)Shuffle data: set epoch of distributed (or fairstore) sampler.
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            # Resume from the iteration if valid
            if hasattr(sampler, "set_start_iter"):
                if (
                    compute_start_iter
                    and self.checkpoint is not None
                    and self.checkpoint["iteration"] > 0
                ):
                    num_iters_in_epochs = len(self.dataloaders[phase_type])
                    num_epochs = self.checkpoint["train_phase_idx"] + 1
                    num_train_iters_done = num_epochs * num_iters_in_epochs
                    start_iter = self.checkpoint["iteration"] - num_train_iters_done
                else:
                    start_iter = 0
                sampler.set_start_iter(start_iter)
            print_sampler_config(sampler)
        # delete the old data iterator
        del self.data_iterator
        gc.collect()
        # recreate the data iterator
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
            phase_type, epoch=self.phase_idx + 1, compute_start_iter=True
        )

        # set the model to train or eval depending on what phase we are in
        self.base_model.train(phase["train"])

        if self.train and self.train_phase_idx >= 0:
            self.optimizer.on_epoch(self.where)

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
            logging.info("Broadcast model BN buffers from master on every forward pass")
            broadcast_buffers_enum_mode = BroadcastBuffersMode.FORWARD_PASS
            self.set_distributed_options(
                broadcast_buffers_mode=broadcast_buffers_enum_mode
            )  # NOQA

    def run_hooks(self, hook_function_name):
        """
        Override the ClassyTask run_hook function and run the hooks whenever called
        """
        for hook in self.hooks:
            getattr(hook, hook_function_name)(self)

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
        self.dataloaders = self.build_dataloaders(pin_memory=pin_memory)
        self.phases = self._build_phases()
        train_phases = [phase for phase in self.phases if phase["train"]]
        num_train_phases = len(train_phases)
        self.base_model = self._build_model()
        self._set_ddp_options()
        self.base_loss = self._build_loss()
        self.meters = self._build_meters()
        self.optimizer = self._build_optimizer()
        self.optimizer_schedulers = self._build_optimizer_schedulers()
        self.num_train_phases = num_train_phases

        self.base_loss = self.base_loss.to(self.device)
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

        # Restore an hypothetical checkpoint
        vissl_state_dict = None
        if self.checkpoint_path is not None:
            self.checkpoint = load_and_broadcast_checkpoint(
                checkpoint_path=self.checkpoint_path, device=torch.device("cpu")
            )
            self.iteration = self.checkpoint["iteration"]
            self.local_iteration_num = self.checkpoint["iteration_num"]
            vissl_state_dict = self.checkpoint.get("classy_state_dict")
            if "loss" in self.checkpoint:
                self.base_loss.load_state_dict(self.checkpoint["loss"])
                logging.info("======Loaded loss state from checkpoint======")

        return self._update_classy_state(vissl_state_dict)

    def prepare_extraction(self, pin_memory: bool = False):
        """
        Prepares a light-weight task for feature extraction on multi-gpu. The model
        runs in eval mode only.
        """
        self.dataloaders = self.build_dataloaders(pin_memory=pin_memory)
        self.base_model = self._build_model()
        if self.device.type == "cuda":
            self.base_model = copy_model_to_gpu(self.base_model)
        return self

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
