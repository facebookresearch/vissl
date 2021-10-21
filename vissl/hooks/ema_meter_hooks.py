import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.hooks.log_hooks import LogLossLrEtaHook
from classy_vision.generic.distributed_util import is_primary

class EmaMetersHook(ClassyHook):
    """
    Hook executed for EMA model
    """
    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_backward = ClassyHook._noop

    def on_start(self, task: "tasks.ClassyTask") -> None:
        task.ema_meters = task._build_meters()

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        # reset the meters at the beginning of the epoch
        for meter in task.ema_meters:
            meter.reset()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Update ema model weights
        """
        if task.iteration % task.config.HOOKS.EMA_METERS.UPDATE_ITER != 0:
            return

        if task.where < task.config.HOOKS.EMA_METERS.WARMUP:
            task.ema_model.set(task.base_model)
        else:
            assert (
                task.ema_model.decay >= 0 and task.ema_model.decay <= 1
            ), f"Decay is {task.ema_model.decay}"
            task.ema_model.update(task.base_model)


    def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
        """
        Sync states between the meters if running distributed training
        and log the meters.
        """
        # WARNING: This will not work correctly with gradient accumulation
        with torch.no_grad():
            model_output = task.ema_model.module(task.last_batch.sample['input'])
            target = task.last_batch.sample["target"]
            for meter in task.ema_meters:
                meter.update(model_output, target.detach())

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        for meter in task.ema_meters:
            meter.sync_state()

        if is_primary():
            LogLossLrEtaHook.print_and_save_meters(
                task,
                task.train_phase_idx,
                task.ema_meters,
                metric_key_name_suffix="ema"
            )
