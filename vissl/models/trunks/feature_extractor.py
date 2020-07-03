# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from vissl.models.model_helpers import Identity
from vissl.models.trunks import TRUNKS


POOL_OPS = {
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
    "MaxPool2d": nn.MaxPool2d,
    "Identity": Identity,
}


class FeatureExtractorModel(nn.Module):
    def __init__(self, model_config):
        super(FeatureExtractorModel, self).__init__()

        self.model_config = model_config
        trunk_name = model_config["TRUNK"]["NAME"]
        assert trunk_name in TRUNKS, "Trunk unknown"
        self.base_model = TRUNKS[trunk_name](self.model_config, trunk_name)
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
            if self.model_config.TRUNK.SHOULD_FLATTEN:
                feat = torch.flatten(feat, start_dim=1)
            out.append(feat)
        return out

    def _freeze_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _attach_feature_pool_layers(self):
        feat_pool = nn.ModuleList(
            [
                POOL_OPS[pool_ops](*args)
                for (pool_ops, args) in self.model_config.TRUNK.LINEAR_FEAT_POOL_OPS
            ]
        )
        return feat_pool

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.base_model.eval()
        return self
