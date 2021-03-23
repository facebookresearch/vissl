from collections import defaultdict

from fvcore.common.history_buffer import HistoryBuffer


_VISSL_EVENT_STORAGE_STACK = []


def get_event_storage():
    return _VISSL_EVENT_STORAGE_STACK[-1]


def create_event_storage():
    _VISSL_EVENT_STORAGE_STACK.append(VisslEventStorage())


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
        # self._vis_data = []     # later for tensorboard
        # self._histograms = []   # later for tensorboard

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
        return NotImplementedError

    def clear_histograms(self):
        return NotImplementedError

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        return NotImplementedError

    def put_image(self, img_name, img_tensor):
        # implement later for tensorboard
        return NotImplementedError
