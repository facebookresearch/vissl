# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed like attributes (as well as normally).
    Credits: https://aiida.readthedocs.io/projects/aiida-core/en/latest/_modules/aiida/common/extendeddicts.html#AttributeDict  # noqa
    """

    def __init__(self, dictionary):
        """
        Recursively turn the `dict` and all its nested dictionaries into `AttrDict` instance.
        """
        super().__init__()

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            else:
                self[key] = value

    def to_dict(self):
        """
        Convert the AttrDict back to a dictionary.

        Helpful to feed the configuration to generic functions
        which only accept primitive types
        """
        dict = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                dict[k] = v.to_dict()
            else:
                dict[k] = v
        return dict

    def __getattr__(self, key):
        """
        Read a key as an attribute.

        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            return self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __setattr__(self, key, value):
        """
        Set a key as an attribute.
        """
        self[key] = value

    def __delattr__(self, key):
        """
        Delete a key as an attribute.

        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            del self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __getstate__(self):
        """
        Needed for pickling this class.
        """
        return self.__dict__.copy()

    def __setstate__(self, dictionary):
        """
        Needed for pickling this class.
        """
        self.__dict__.update(dictionary)

    def __deepcopy__(self, memo=None):
        """
        Deep copy.
        """
        from copy import deepcopy

        if memo is None:
            memo = {}
        retval = deepcopy(dict(self))
        return self.__class__(retval)

    def __dir__(self):
        return self.keys()
