#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import gc
import logging
import os

import torch
from classy_vision.generic.util import copy_model_to_gpu
from classy_vision.losses import build_loss
from classy_vision.meters import build_meter
from classy_vision.optim import build_optimizer
from classy_vision.tasks import ClassificationTask, register_task
from classy_vision.tasks.classification_task import BroadcastBuffersMode
from vissl.dataset import build_dataset, get_loader
from vissl.models import build_model, convert_sync_bn
from vissl.utils.checkpoint import init_model_from_weights
from vissl.utils.misc import is_apex_available


if is_apex_available():
    import apex


@register_task("self_supervision_task")
class SelfSupervisionTask(ClassificationTask):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.checkpoint = None
        self.available_splits = []
        self.loss = None
        self.meters = None
        self.datasets = None
        self.phases = []
        self.hooks = []
        self.base_model = None
        self.optimizer = None
        self.amp_args = None
        self.data_and_label_keys = []
        self.set_amp_args()
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

    def set_available_splits(self):
        # self.available_splits = list(self.config["DATA"].keys())
        if self.config.TEST_MODEL:
            self.available_splits.append("TEST")
        if not self.config.TEST_ONLY:
            self.available_splits.append("TRAIN")
        return self

    def set_amp_args(self):
        """
        amp_args is a dictionary containing arguments to be passed to
        amp.initialize. Set to None to disable amp.  To enable mixed
        precision training, pass amp_args={"opt_level": "O1"} here.
        See https://nvidia.github.io/apex/amp.html for more info.
        """
        amp_args = None
        if self.config.MODEL.AMP_PARAMS.USE_AMP:
            if not is_apex_available():
                raise RuntimeError("Apex is not available. Can't use mixed precision")
            amp_args = self.config.MODEL.AMP_PARAMS.AMP_ARGS
        self.amp_args = amp_args
        logging.info(f"Setting amp args: {self.amp_args}")

    def set_checkpoint(self, checkpoint):
        assert (
            checkpoint is None or "classy_state_dict" in checkpoint
        ), "Checkpoint does not contain classy_state_dict"
        self.checkpoint = checkpoint

    def set_iteration(self, iteration):
        assert iteration >= 0, "Iteration number must be positive"
        self.iteration = iteration

    @classmethod
    def from_config(cls, config):
        test_only = config.TEST_ONLY
        return cls(config).set_available_splits().set_test_only(test_only)

    # We keep the function because this is used by hooks like checkpoint etc.
    def get_config(self):
        return {"config": self.config}

    def _build_phases(self):
        """
        Returns list of phases from config.  These phases will look like:
        {
          train: is this a train or test phase (bool)?
        }
        If this is a test only run, then only test phases will be
        generated, if this is a training run, then x phases = x train
        phases + x test phases, interleaved. We also add the test phases
        every TEST_EVERY_NUM_EPOCH if we don't want the tst to run after every test
        phase.
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
        datasets, data_and_label_keys = {}, {}
        for split in self.available_splits:
            datasets[split] = build_dataset(self.config, split)
            data_and_label_keys["input"] = self.config.DATA[split].INPUT_KEY_NAMES
            data_and_label_keys["target"] = self.config.DATA[split].TARGET_KEY_NAMES
        return datasets, data_and_label_keys

    def build_dataloaders(self, pin_memory):
        self.datasets, self.data_and_label_keys = self.build_datasets()
        loaders = {}
        for split in self.available_splits:
            loader = get_loader(
                dataset=self.datasets[split],
                dataset_config=self.config["DATA"][split],
                num_dataloader_workers=self.config.MACHINE.NUM_DATALOADER_WORKERS,
                pin_memory=pin_memory,
            )
            loaders[split.lower()] = loader
        return loaders

    def _build_optimizer(self):
        optimizer_config = self.config["OPTIMIZER"]
        if optimizer_config.use_larc:
            assert is_apex_available(), "Apex must be available to use LARC"
        optim = build_optimizer(optimizer_config)
        return optim

    def _build_loss(self):
        # in some cases like memory bank, we need to store the size of data
        # as we use it to allocate memory. Hence we set that parameter here.
        logging.info("Building loss...")
        for split in self.available_splits:
            if split == "TRAIN":
                self.config["CRITERION"]["NUM_TRAIN_SAMPLES"] = len(
                    self.datasets["TRAIN"]
                )
            if split == "TEST":
                self.config["CRITERION"]["NUM_TRAIN_SAMPLES"] = len(
                    self.datasets["TEST"]
                )
        loss = build_loss(self.config["CRITERION"])
        return loss

    def _build_meters(self):
        """
        Returns meters for task.
        """
        meters = self.config.get("METERS", None)
        if meters is None:
            meters = {}
        meter_items = meters.items()
        meter_configs = [{"name": name, **args} for name, args in meter_items]
        return [build_meter(config) for config in meter_configs]

    def _restore_model_weights(self, model):
        params_from_file = self.config["MODEL"]["PARAMS_FILE"]
        init_weights_path = params_from_file["PATH"]
        logging.info(f"Initializing model from: {init_weights_path}")

        if os.path.exists(init_weights_path):
            weights = torch.load(init_weights_path, map_location=torch.device("cpu"))
            skip_layers = params_from_file.get("SKIP_LAYERS", None)
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
        Returns model for task.
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
        if self.config["MODEL"]["FEATURE_EVAL_MODE"]:
            model.freeze_trunk()

        # If we want to initialize the model in case of finetuning or evaluation,
        # we do it here. But we check that there is no checkpoint existing before
        # This is important in cases when the model training dies.
        if self.checkpoint is None and os.path.exists(
            self.config["MODEL"]["PARAMS_FILE"]["PATH"]
        ):
            model = self._restore_model_weights(model)

        return model

    def recreate_data_iterator(self, phase_type):
        # re-create the data iterator
        del self.data_iterator
        gc.collect()
        self.data_iterator = iter(self.dataloaders[phase_type])

    def _set_classy_state(self, device, state):
        """
        We overwrite the classy state setting here to match our dataloader calls
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
        if device == "gpu":
            self.base_model.cuda()
        for meter, meter_state in zip(self.meters, state["meters"]):
            meter.set_classy_state(meter_state)
        self.optimizer.set_classy_state(state["optimizer"])

        # restore amp state. It's called after amp.initialize is done.
        if "amp" in state:
            if is_apex_available():
                apex.amp.load_state_dict(state["amp"])
            else:
                logging.warning(
                    "Loading a checkpoint which has amp state but apex isn't available now"
                )

        self.phase_idx = state["phase_idx"]
        self.train_phase_idx = state["train_phase_idx"]
        self.num_updates = state["num_updates"]
        self.losses = state["losses"]

        phase_type = "train" if self.train else "test"
        phase = self.phases[self.phase_idx]

        # re-create the data iterator
        self.recreate_data_iterator(phase_type)

        # set the model to train or eval depending on what phase we are in
        self.base_model.train(phase["train"])

        # update the optimizer
        # Here, the state.num_updates is used to calculate where we are in the
        # training. If we are resuming, the state.num_updates will be restored
        # and used to calculate the proper LR to use
        if self.train and self.train_phase_idx >= 0:
            self.optimizer.update_schedule_on_epoch(self.where)

    def _update_classy_state(self, device, state_dict=None):
        """
        Updates classy state with the provided state dict from a checkpoint.
        state_dict = checkpoint loaded state
        """
        if state_dict is not None:
            try:
                self._set_classy_state(device, state_dict)
                success = True
            except Exception as e:
                logging.exception(f"Could not load the checkpoint: {e}")
                success = False
            assert success, "Update classy state from checkpoint failed."
        return self

    def _set_ddp_options(self):
        # set DDP options if the user has supplied them
        broadcast_buffers = self.config["DISTRIBUTED"]["BROADCAST_BUFFERS"]
        if broadcast_buffers:
            logging.info("Broadcast model BN buffers from master on every forward pass")
            broadcast_buffers_enum_mode = BroadcastBuffersMode.FORWARD_PASS
            self.set_distributed_options(
                broadcast_buffers_mode=broadcast_buffers_enum_mode
            )  # NOQA

    # override the ClassyTask run_hook function
    def run_hooks(self, hook_function_name):
        for hook in self.hooks:
            getattr(hook, hook_function_name)(self)

    def prepare(self, device: str, pin_memory: bool = False):
        """
        Prepares the task.
        """
        self.dataloaders = self.build_dataloaders(pin_memory=pin_memory)
        self.phases = self._build_phases()
        train_phases = [phase for phase in self.phases if phase["train"]]
        num_train_phases = len(train_phases)
        self.base_model = self._build_model()
        self._set_ddp_options()
        self.loss = self._build_loss()
        self.meters = self._build_meters()
        self.optimizer = self._build_optimizer()
        self.iteration = self.iteration
        self.num_train_phases = num_train_phases

        if self.use_gpu:
            self.loss = self.loss.cuda()
            self.base_model = copy_model_to_gpu(self.base_model)

        # initialize the pytorch optimizer now since the model has been moved to
        # the appropriate device
        self.prepare_optimizer(
            optimizer=self.optimizer, model=self.base_model, loss=self.loss
        )
        if self.amp_args is not None and is_apex_available():
            # Allow Amp to perform casts as specified by the amp_args.
            # This updates the model and the PyTorch optimizer (which is wrapped
            # by the ClassyOptimizer in self.optimizer).
            # NOTE: this must happen before loading the checkpoint. See
            # https://nvidia.github.io/apex/amp.html#checkpointing for more details.
            self.base_model, self.optimizer.optimizer = apex.amp.initialize(
                self.base_model, self.optimizer.optimizer, **self.amp_args
            )

        vissl_state_dict = None
        if self.checkpoint is not None:
            self.iteration = self.checkpoint["iteration"]
            self.local_iteration_num = self.checkpoint["iteration_num"]
            vissl_state_dict = self.checkpoint.get("classy_state_dict")
            if "criterion" in self.checkpoint:
                self.loss.load_state_dict(self.checkpoint["criterion"])
                logging.info("======Loaded criterion state from checkpoint======")

        return self._update_classy_state(device, vissl_state_dict)

    def prepare_extraction(self, device, pin_memory=False, use_gpu=True):
        """
        Prepares a light-weight task for feature extraction on multi-gpu. The model
        runs in eval mode only.
        """
        self.dataloaders = self.build_dataloaders(pin_memory=pin_memory)
        self.base_model = self._build_model()
        if use_gpu:
            self.base_model = copy_model_to_gpu(self.base_model)
        return self
