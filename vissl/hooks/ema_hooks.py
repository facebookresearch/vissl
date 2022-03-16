import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.hooks.log_hooks import LogLossMetricsCheckpointHook


class EmaHook(ClassyHook):
    """
    Hook executed to save the exponential moving average of the base_model parameters
    and run the meters on this ema_model.
    """

    on_phase_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_backward = ClassyHook._noop

    def __init__(self, enable_ema_meters: bool, update_iter: int, warmup: float):
        super().__init__()
        self.enable_ema_meters = enable_ema_meters
        self.update_iter = update_iter
        self.warmup = warmup

    def on_start(self, task: "tasks.ClassyTask") -> None:
        if self.enable_ema_meters:
            task.ema_meters = task._build_meters()

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        # reset the meters at the beginning of the epoch
        for meter in task.ema_meters:
            meter.reset()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Update ema model weights
        """
        if task.iteration % self.update_iter != 0:
            return

        if task.where < self.warmup:
            # Overwrite model weights.
            task.ema_model.set(task.base_model)
        else:
            assert (
                task.ema_model.decay >= 0 and task.ema_model.decay <= 1
            ), f"Decay is {task.ema_model.decay}"
            task.ema_model.update(task.base_model)

    @torch.no_grad()
    def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
        """
        Sync states between the meters if running distributed training
        and log the meters.
        """
        if not self.enable_ema_meters:
            # Do not process the ema meters, merely save the EMA model
            return

        model_input = task.last_batch.sample["input"].to(task.ema_model.device)
        model_output = task.ema_model.module(model_input)

        if isinstance(model_output, list):
            model_output_cpu = [x.cpu() for x in model_output]
        else:
            model_output_cpu = model_output.cpu()

        target = task.last_batch.sample["target"]

        for meter in task.ema_meters:
            meter.update(model_output_cpu, target.detach().cpu())

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        for meter in task.ema_meters:
            meter.sync_state()

        if is_primary():
            LogLossMetricsCheckpointHook.print_and_save_meters(
                task,
                task.train_phase_idx,
                task.ema_meters,
                metric_key_name_suffix="ema",
            )
