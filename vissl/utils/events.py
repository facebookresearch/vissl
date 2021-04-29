import json
from collections import defaultdict

import torch
from fvcore.common.file_io import PathManager
from fvcore.common.history_buffer import HistoryBuffer


_VISSL_EVENT_STORAGE_STACK = []


def get_event_storage():
    return _VISSL_EVENT_STORAGE_STACK[-1]


def create_event_storage():
    _VISSL_EVENT_STORAGE_STACK.append(VisslEventStorage(1))


class VisslEventWriter:
    """
    Base class for writers that obtain events from :class:`VisslEventStorage`
    and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class VisslEventStorage:
    """
    The user-facing class that stores the running metrics
    and all the logging data in one place. The Storage can
    be updated anywhere in the training step. The storage
    is used by several writers/hooks to write the data to
    several backends (json, tensorboard, etc)
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._latest_scalars = {}
        self._iter = start_iter
        self._vis_data = []  # later for tensorboard
        self._histograms = []  # later for tensorboard

    def put_scalar(self, name, value):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        """
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = (value, self._iter)

    def put_scalars(self, **kwargs):
        """
        Put multiple scalars from keyword arguments.
        Examples:
            storage.put_scalars(loss=my_loss, accuracy=my_accuracy)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError(f"No history metric available for {name}!")
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    def clear_images(self):
        self._vis_data = []

    def clear_histograms(self):
        self._histograms = []

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create a histogram from a tensor.
        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(
            start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32
        )

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = {
            "tag": hist_name,
            "min": ht_min,
            "max": ht_max,
            "num": len(hist_tensor),
            "sum": float(hist_tensor.sum()),
            "sum_squares": float(torch.sum(hist_tensor ** 2)),
            "bucket_limits": hist_edges[1:].tolist(),
            "bucket_counts": hist_counts.tolist(),
            "global_step": self._iter,
        }
        self._histograms.append(hist_params)

    def put_image(self, img_name, img_tensor):
        # implement later for tensorboard
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.
        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))


class JsonWriter(VisslEventWriter):
    def __init__(self, json_file):
        """
        Args:
            json_file: path to the json file. New data will be appended if the file
                       exists.
        """
        self._file_handle = PathManager.open(json_file, "w")

    def write(self):
        storage: VisslEventStorage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest().items():
            # keep scalars that have not been written
            to_save[iter][k] = v
        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()

    def close(self):
        self._file_handle.close()


class TensorboardWriter(VisslEventWriter):
    def __init__(self, log_dir: str, flush_secs: int, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            flush_secs (int): flush data to tensorboard every flush_secs
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._flush_secs = flush_secs
        from torch.utils.tensorboard import SummaryWriter

        self._tb_writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._tb_writer.add_image(img_name, img, step_num)

            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._tb_writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        if hasattr(self, "_tb_writer"):  # doesn't exist when the code fails at import
            self._writer.close()
