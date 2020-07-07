# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class VisslPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="vissl", path="pkg://configs")
