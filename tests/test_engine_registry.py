# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from vissl.engines.engine_registry import Engine, get_engine, register_engine


class TestEngineRegistry(unittest.TestCase):
    """
    Test that the engine registry allows to:
    - select the right engine based on the configuration
    - deal with the typical error cases (unknowns, duplicates, etc)
    """

    def test_valid_engines(self) -> None:
        engine = get_engine("train")
        assert engine is not None
        assert isinstance(engine, Engine)

        engine = get_engine("extract_features")
        assert engine is not None
        assert isinstance(engine, Engine)

    def test_unknown_engine_raises_error(self) -> None:
        with self.assertRaises(ValueError) as result:
            get_engine("unknown_name")
        msg = str(result.exception)
        assert msg.startswith("Unknown engine name unknown_name")

    def test_duplicate_engine_registration_error(self) -> None:
        with self.assertRaises(ValueError) as result:

            @register_engine("train")
            class Duplicate(Engine):
                def run_engine(self, *args, **kwargs):
                    pass

        msg = str(result.exception)
        assert msg.startswith("Engine (train) already registered at")
