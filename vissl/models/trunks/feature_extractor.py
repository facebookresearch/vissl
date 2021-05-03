# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import Identity
from vissl.models.trunks import get_model_trunk


POOL_OPS = {
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
    "MaxPool2d": nn.MaxPool2d,
    "Identity": Identity,
}


class FeatureExtractorModel(nn.Module):
    def __init__(self, model_config: AttrDict):
        super(FeatureExtractorModel, self).__init__()
        logging.info("Creating Feature extractor trunk...")
        self.model_config = model_config
        trunk_name = model_config["TRUNK"]["NAME"]
        self.base_model = get_model_trunk(trunk_name)(self.model_config, trunk_name)
        self.feature_pool_ops = self._attach_feature_pool_layers()
        self._freeze_model()

    def forward(self, batch, out_feat_keys):
        feats = self.base_model(batch, out_feat_keys)
        assert len(feats) == len(
            self.feature_pool_ops
        ), "#features returned by base model ({}) != #Pooling Ops ({})".format(
            len(feats), len(self.feature_pool_ops)
        )
        out = []
        for feat, op in zip(feats, self.feature_pool_ops):
            feat = op(feat)
            if self.model_config.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS:
                feat = torch.flatten(feat, start_dim=1)
            out.append(feat)
        return out

    def _freeze_model(self):
        logging.info("Freezing model trunk...")
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _attach_feature_pool_layers(self):
        feat_pool_ops = []
        for (
            item
        ) in self.model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP:
            pool_ops, args = item[1]
            feat_pool_ops.append(POOL_OPS[pool_ops](*args))
        feat_pool = nn.ModuleList(feat_pool_ops)
        return feat_pool

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.base_model.eval()
        return self
