# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import traceback
from typing import Any, Callable, List

from classy_vision.hooks import ClassyHook
from vissl.config.attr_dict import AttrDict
from vissl.hooks import default_hook_generator


class Engine(abc.ABC):
    """
    Abtract class for all engines that can be registered in VISSL
    """

    @abc.abstractmethod
    def run_engine(
        self,
        cfg: AttrDict,
        dist_run_id: str,
        checkpoint_path: str,
        checkpoint_folder: str,
        local_rank: int = 0,
        node_id: int = 0,
        hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
    ): ...


_ENGINE_REGISTRY = {}
_ENGINE_REGISTRY_TRACE_BACK = {}


def register_engine(name: str):
    """
    Register an engine

    The decorator allows to register specific kind of actions (like
    training a model, extracting its feature, etc) that can be run
    by VISSL by providing the corresponding name in the 'engine_name'
    of the configuration.
    """

    def register_engine_function(cls):
        if name in _ENGINE_REGISTRY:
            tb = _ENGINE_REGISTRY_TRACE_BACK[name]
            msg = f"Engine ({name}) already registered at \n{tb}\n"
            raise ValueError(msg)

        if not issubclass(cls, Engine):
            raise ValueError(f"Engine ({name}: {cls.__name__}) must extend Engine")

        _ENGINE_REGISTRY[name] = cls
        _ENGINE_REGISTRY_TRACE_BACK[name] = "".join(traceback.format_stack())
        return cls

    return register_engine_function


def get_engine(engine_name: str) -> Engine:
    if engine_name not in _ENGINE_REGISTRY:
        valid_names = ",".join(_ENGINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown engine name {engine_name}: please use any of [{valid_names}]"
        )

    cls = _ENGINE_REGISTRY[engine_name]
    return cls()


def run_engine(
    engine_name: str,
    cfg: AttrDict,
    dist_run_id: str,
    checkpoint_path: str,
    checkpoint_folder: str,
    local_rank: int = 0,
    node_id: int = 0,
    hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
):
    engine = get_engine(engine_name)
    engine.run_engine(
        cfg,
        dist_run_id,
        checkpoint_path,
        checkpoint_folder,
        local_rank=local_rank,
        node_id=node_id,
        hook_generator=hook_generator,
    )
