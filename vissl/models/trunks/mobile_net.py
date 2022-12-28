# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mobilenetv3 import mobilenetv3_large_100
from torchvision.models import mobilenet_v3_large as tv_mobilenet_v3_large
from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk


"""
-------------------------------------------------------------------------------
MobileNet-V3 (Torchvision Implementation)
-------------------------------------------------------------------------------
"""


@register_model_trunk("mobilenetv3_tv")
class MobileNetV3_TV(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.mobile_net_config = model_config.TRUNK.MOBILE_NET
        self.name = self.mobile_net_config.NAME
        self.pretrained = self.mobile_net_config.get("PRETRAINED", False)

        # TIMM models are trained with these BN settings
        self.timm_bn = self.mobile_net_config.get("TIMM_BN", False)
        if self.timm_bn:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
        else:
            norm_layer = None

        if self.name == "mobilenetv3_large_100":
            self.model = tv_mobilenet_v3_large(
                pretrained=self.pretrained, norm_layer=norm_layer
            )
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {self.mobile_net_config.NAME}")

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            return self.forward_features(x)
        else:
            return self.get_intermediate_features(x, names=out_feat_keys)

    def forward_features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return [x]

    def get_intermediate_features(self, x, names: List[str]) -> List[torch.Tensor]:
        trunk_out = self.model.features(x)
        trunk_pool = self.model.avgpool(trunk_out)
        trunk_pool = torch.flatten(trunk_pool, start_dim=1)
        assert len(trunk_pool.shape) == 2
        trunk_pool_norm = F.normalize(trunk_pool, dim=-1)

        outputs = []
        for feat_name in names:
            if feat_name == "trunk_pool":
                outputs.append(trunk_pool)
            elif feat_name == "trunk_pool_norm":
                outputs.append(trunk_pool_norm)
            elif feat_name == "trunk":
                outputs.append(trunk_out)
        return outputs


def download_mobilenet_v3_torchvision_weights(output_path: str):
    """
    Download the Torchvision weights for MobileNet-V3 and save
    them in VISSL compatible format
    """

    # Download the weights from torchvision
    tv_model = tv_mobilenet_v3_large(pretrained=True)

    # Separate the weights in head and trunk for VISSL
    trunk = {}
    heads = {}
    for k, v in tv_model.state_dict().items():
        if k.startswith("classifier"):
            heads[f"0.{k}"] = v
        else:
            k = f"model.{k}"
            trunk[k] = v

    # Save the checkpoint in VISSL format
    out_cp = {
        "classy_state_dict": {
            "base_model": {
                "model": {
                    "trunk": trunk,
                    "heads": heads,
                }
            }
        }
    }
    torch.save(out_cp, output_path)


"""
-------------------------------------------------------------------------------
MobileNet-V3 (Timm Implementation)
-------------------------------------------------------------------------------
"""


@register_model_trunk("mobilenetv3_timm")
class MobileNetV3_Timm(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.mobile_net_config = model_config.TRUNK.MOBILE_NET
        self.name = self.mobile_net_config.NAME
        self.pretrained = self.mobile_net_config.get("PRETRAINED", False)
        self.trunk_only = self.mobile_net_config.get("TRUNK_ONLY", True)
        if self.name == "mobilenetv3_large_100":
            self.model = mobilenetv3_large_100(pretrained=self.pretrained)
            if self.trunk_only:
                self.model.conv_head = nn.Identity()
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {self.mobile_net_config.NAME}")

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            return self.forward_features(x)
        else:
            return self.get_intermediate_features(x, names=out_feat_keys)

    def forward_features(self, x):
        x = self.model.forward_features(x)
        if self.trunk_only:
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            x = x.flatten(start_dim=1)
            return [x]

        x = self.model.forward_head(x, pre_logits=True)
        return [x]

    def get_intermediate_features(self, x, names: List[str]) -> List[torch.Tensor]:
        trunk_out = self.model.forward_features(x)
        head_out = self.model.forward_head(trunk_out, pre_logits=True)

        trunk_pool = F.adaptive_avg_pool2d(trunk_out, output_size=(1, 1))
        trunk_pool = torch.flatten(trunk_pool, start_dim=1)
        trunk_pool_norm = F.normalize(trunk_pool, dim=-1)
        head_out_norm = F.normalize(head_out, dim=-1)

        outputs = []
        for feat_name in names:
            if feat_name == "flatten":
                outputs.append(head_out)
            elif feat_name == "flatten_norm":
                outputs.append(head_out_norm)
            elif feat_name == "trunk_pool":
                outputs.append(trunk_pool)
            elif feat_name == "trunk_pool_norm":
                outputs.append(trunk_pool_norm)
            elif feat_name == "trunk":
                outputs.append(trunk_out)
        return outputs


def download_mobilenet_v3_timm_weights(output_path: str, to_torchvision: bool = True):
    timm_model = mobilenetv3_large_100(pretrained=True)

    trunk = {}
    heads = {}
    for k, v in timm_model.state_dict().items():
        if k.startswith("conv_head"):
            if k == "conv_head.weight":
                heads["0.classifier.0.weight"] = v
                if to_torchvision:
                    heads["0.classifier.0.weight"] = v.squeeze(-1).squeeze(-1)
                else:
                    heads["0.classifier.0.weight"] = v
            else:
                heads["0.classifier.0.bias"] = v
        elif k.startswith("classifier"):
            if to_torchvision:
                if k == "classifier.weight":
                    heads["0.classifier.3.weight"] = v
                else:
                    heads["0.classifier.3.bias"] = v
            else:
                if k == "classifier.weight":
                    heads["0.classifier.4.weight"] = v
                else:
                    heads["0.classifier.4.bias"] = v
        else:
            trunk[f"model.{k}"] = v

    if to_torchvision:
        trunk = ConvertorTimmTV.convert(trunk)

    out_cp = {
        "classy_state_dict": {
            "base_model": {
                "model": {
                    "trunk": trunk,
                    "heads": heads,
                }
            }
        }
    }
    torch.save(out_cp, output_path)


"""
-------------------------------------------------------------------------------
Weight conversion between Torchvision and Timm
-------------------------------------------------------------------------------
"""


class ConvertorTimmTV:
    @classmethod
    def convert(cls, state_dict):
        out_state_dict = {}
        for tv_name, timm_name in cls.MAPPING:
            out_state_dict[tv_name] = state_dict[timm_name]
        return out_state_dict

    MAPPING = [
        ("model.features.0.0.weight", "model.conv_stem.weight"),
        ("model.features.0.1.weight", "model.bn1.weight"),
        ("model.features.0.1.bias", "model.bn1.bias"),
        ("model.features.0.1.running_mean", "model.bn1.running_mean"),
        ("model.features.0.1.running_var", "model.bn1.running_var"),
        ("model.features.0.1.num_batches_tracked", "model.bn1.num_batches_tracked"),
        ("model.features.1.block.0.0.weight", "model.blocks.0.0.conv_dw.weight"),
        ("model.features.1.block.0.1.weight", "model.blocks.0.0.bn1.weight"),
        ("model.features.1.block.1.1.weight", "model.blocks.0.0.bn2.weight"),
        ("model.features.1.block.0.1.bias", "model.blocks.0.0.bn1.bias"),
        ("model.features.1.block.1.1.bias", "model.blocks.0.0.bn2.bias"),
        (
            "model.features.1.block.0.1.running_mean",
            "model.blocks.0.0.bn1.running_mean",
        ),
        (
            "model.features.1.block.1.1.running_mean",
            "model.blocks.0.0.bn2.running_mean",
        ),
        ("model.features.1.block.0.1.running_var", "model.blocks.0.0.bn1.running_var"),
        ("model.features.1.block.1.1.running_var", "model.blocks.0.0.bn2.running_var"),
        (
            "model.features.1.block.0.1.num_batches_tracked",
            "model.blocks.0.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.1.block.1.1.num_batches_tracked",
            "model.blocks.0.0.bn2.num_batches_tracked",
        ),
        ("model.features.1.block.1.0.weight", "model.blocks.0.0.conv_pw.weight"),
        ("model.features.2.block.0.0.weight", "model.blocks.1.0.conv_pw.weight"),
        ("model.features.2.block.0.1.weight", "model.blocks.1.0.bn1.weight"),
        ("model.features.2.block.1.1.weight", "model.blocks.1.0.bn2.weight"),
        ("model.features.2.block.0.1.bias", "model.blocks.1.0.bn1.bias"),
        ("model.features.2.block.1.1.bias", "model.blocks.1.0.bn2.bias"),
        (
            "model.features.2.block.0.1.running_mean",
            "model.blocks.1.0.bn1.running_mean",
        ),
        (
            "model.features.2.block.1.1.running_mean",
            "model.blocks.1.0.bn2.running_mean",
        ),
        ("model.features.2.block.0.1.running_var", "model.blocks.1.0.bn1.running_var"),
        ("model.features.2.block.1.1.running_var", "model.blocks.1.0.bn2.running_var"),
        (
            "model.features.2.block.0.1.num_batches_tracked",
            "model.blocks.1.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.2.block.1.1.num_batches_tracked",
            "model.blocks.1.0.bn2.num_batches_tracked",
        ),
        (
            "model.features.2.block.2.1.num_batches_tracked",
            "model.blocks.1.0.bn3.num_batches_tracked",
        ),
        ("model.features.2.block.1.0.weight", "model.blocks.1.0.conv_dw.weight"),
        ("model.features.2.block.2.0.weight", "model.blocks.1.0.conv_pwl.weight"),
        ("model.features.2.block.2.1.weight", "model.blocks.1.0.bn3.weight"),
        ("model.features.2.block.2.1.bias", "model.blocks.1.0.bn3.bias"),
        (
            "model.features.2.block.2.1.running_mean",
            "model.blocks.1.0.bn3.running_mean",
        ),
        ("model.features.2.block.2.1.running_var", "model.blocks.1.0.bn3.running_var"),
        ("model.features.3.block.0.0.weight", "model.blocks.1.1.conv_pw.weight"),
        ("model.features.3.block.0.1.weight", "model.blocks.1.1.bn1.weight"),
        ("model.features.3.block.1.1.weight", "model.blocks.1.1.bn2.weight"),
        ("model.features.3.block.0.1.bias", "model.blocks.1.1.bn1.bias"),
        ("model.features.3.block.1.1.bias", "model.blocks.1.1.bn2.bias"),
        (
            "model.features.3.block.0.1.running_mean",
            "model.blocks.1.1.bn1.running_mean",
        ),
        (
            "model.features.3.block.1.1.running_mean",
            "model.blocks.1.1.bn2.running_mean",
        ),
        ("model.features.3.block.0.1.running_var", "model.blocks.1.1.bn1.running_var"),
        ("model.features.3.block.1.1.running_var", "model.blocks.1.1.bn2.running_var"),
        (
            "model.features.3.block.0.1.num_batches_tracked",
            "model.blocks.1.1.bn1.num_batches_tracked",
        ),
        (
            "model.features.3.block.1.1.num_batches_tracked",
            "model.blocks.1.1.bn2.num_batches_tracked",
        ),
        (
            "model.features.3.block.2.1.num_batches_tracked",
            "model.blocks.1.1.bn3.num_batches_tracked",
        ),
        ("model.features.3.block.1.0.weight", "model.blocks.1.1.conv_dw.weight"),
        ("model.features.3.block.2.0.weight", "model.blocks.1.1.conv_pwl.weight"),
        ("model.features.3.block.2.1.weight", "model.blocks.1.1.bn3.weight"),
        ("model.features.3.block.2.1.bias", "model.blocks.1.1.bn3.bias"),
        (
            "model.features.3.block.2.1.running_mean",
            "model.blocks.1.1.bn3.running_mean",
        ),
        ("model.features.3.block.2.1.running_var", "model.blocks.1.1.bn3.running_var"),
        ("model.features.4.block.0.0.weight", "model.blocks.2.0.conv_pw.weight"),
        (
            "model.features.4.block.2.fc2.weight",
            "model.blocks.2.0.se.conv_expand.weight",
        ),
        ("model.features.4.block.0.1.weight", "model.blocks.2.0.bn1.weight"),
        ("model.features.4.block.1.1.weight", "model.blocks.2.0.bn2.weight"),
        ("model.features.4.block.0.1.bias", "model.blocks.2.0.bn1.bias"),
        ("model.features.4.block.1.1.bias", "model.blocks.2.0.bn2.bias"),
        ("model.features.4.block.2.fc2.bias", "model.blocks.2.0.se.conv_expand.bias"),
        (
            "model.features.4.block.0.1.running_mean",
            "model.blocks.2.0.bn1.running_mean",
        ),
        (
            "model.features.4.block.1.1.running_mean",
            "model.blocks.2.0.bn2.running_mean",
        ),
        ("model.features.4.block.0.1.running_var", "model.blocks.2.0.bn1.running_var"),
        ("model.features.4.block.1.1.running_var", "model.blocks.2.0.bn2.running_var"),
        (
            "model.features.4.block.0.1.num_batches_tracked",
            "model.blocks.2.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.4.block.1.1.num_batches_tracked",
            "model.blocks.2.0.bn2.num_batches_tracked",
        ),
        (
            "model.features.4.block.3.1.num_batches_tracked",
            "model.blocks.2.0.bn3.num_batches_tracked",
        ),
        ("model.features.4.block.1.0.weight", "model.blocks.2.0.conv_dw.weight"),
        (
            "model.features.4.block.2.fc1.weight",
            "model.blocks.2.0.se.conv_reduce.weight",
        ),
        ("model.features.4.block.2.fc1.bias", "model.blocks.2.0.se.conv_reduce.bias"),
        ("model.features.4.block.3.0.weight", "model.blocks.2.0.conv_pwl.weight"),
        ("model.features.4.block.3.1.weight", "model.blocks.2.0.bn3.weight"),
        ("model.features.4.block.3.1.bias", "model.blocks.2.0.bn3.bias"),
        (
            "model.features.4.block.3.1.running_mean",
            "model.blocks.2.0.bn3.running_mean",
        ),
        ("model.features.4.block.3.1.running_var", "model.blocks.2.0.bn3.running_var"),
        ("model.features.5.block.0.0.weight", "model.blocks.2.1.conv_pw.weight"),
        ("model.features.5.block.0.1.weight", "model.blocks.2.1.bn1.weight"),
        ("model.features.5.block.1.1.weight", "model.blocks.2.1.bn2.weight"),
        ("model.features.5.block.0.1.bias", "model.blocks.2.1.bn1.bias"),
        ("model.features.5.block.1.1.bias", "model.blocks.2.1.bn2.bias"),
        ("model.features.5.block.2.fc2.bias", "model.blocks.2.1.se.conv_expand.bias"),
        (
            "model.features.5.block.0.1.running_mean",
            "model.blocks.2.1.bn1.running_mean",
        ),
        (
            "model.features.5.block.1.1.running_mean",
            "model.blocks.2.1.bn2.running_mean",
        ),
        ("model.features.5.block.0.1.running_var", "model.blocks.2.1.bn1.running_var"),
        ("model.features.5.block.1.1.running_var", "model.blocks.2.1.bn2.running_var"),
        (
            "model.features.5.block.0.1.num_batches_tracked",
            "model.blocks.2.1.bn1.num_batches_tracked",
        ),
        (
            "model.features.5.block.1.1.num_batches_tracked",
            "model.blocks.2.1.bn2.num_batches_tracked",
        ),
        (
            "model.features.5.block.3.1.num_batches_tracked",
            "model.blocks.2.1.bn3.num_batches_tracked",
        ),
        ("model.features.5.block.1.0.weight", "model.blocks.2.1.conv_dw.weight"),
        (
            "model.features.5.block.2.fc1.weight",
            "model.blocks.2.1.se.conv_reduce.weight",
        ),
        ("model.features.5.block.2.fc1.bias", "model.blocks.2.1.se.conv_reduce.bias"),
        (
            "model.features.5.block.2.fc2.weight",
            "model.blocks.2.1.se.conv_expand.weight",
        ),
        ("model.features.5.block.3.0.weight", "model.blocks.2.1.conv_pwl.weight"),
        ("model.features.5.block.3.1.weight", "model.blocks.2.1.bn3.weight"),
        ("model.features.5.block.3.1.bias", "model.blocks.2.1.bn3.bias"),
        (
            "model.features.5.block.3.1.running_mean",
            "model.blocks.2.1.bn3.running_mean",
        ),
        ("model.features.5.block.3.1.running_var", "model.blocks.2.1.bn3.running_var"),
        ("model.features.6.block.0.0.weight", "model.blocks.2.2.conv_pw.weight"),
        ("model.features.6.block.0.1.weight", "model.blocks.2.2.bn1.weight"),
        ("model.features.6.block.1.1.weight", "model.blocks.2.2.bn2.weight"),
        ("model.features.6.block.0.1.bias", "model.blocks.2.2.bn1.bias"),
        ("model.features.6.block.1.1.bias", "model.blocks.2.2.bn2.bias"),
        ("model.features.6.block.2.fc2.bias", "model.blocks.2.2.se.conv_expand.bias"),
        (
            "model.features.6.block.0.1.running_mean",
            "model.blocks.2.2.bn1.running_mean",
        ),
        (
            "model.features.6.block.1.1.running_mean",
            "model.blocks.2.2.bn2.running_mean",
        ),
        ("model.features.6.block.0.1.running_var", "model.blocks.2.2.bn1.running_var"),
        ("model.features.6.block.1.1.running_var", "model.blocks.2.2.bn2.running_var"),
        (
            "model.features.6.block.0.1.num_batches_tracked",
            "model.blocks.2.2.bn1.num_batches_tracked",
        ),
        (
            "model.features.6.block.1.1.num_batches_tracked",
            "model.blocks.2.2.bn2.num_batches_tracked",
        ),
        (
            "model.features.6.block.3.1.num_batches_tracked",
            "model.blocks.2.2.bn3.num_batches_tracked",
        ),
        ("model.features.6.block.1.0.weight", "model.blocks.2.2.conv_dw.weight"),
        (
            "model.features.6.block.2.fc1.weight",
            "model.blocks.2.2.se.conv_reduce.weight",
        ),
        ("model.features.6.block.2.fc1.bias", "model.blocks.2.2.se.conv_reduce.bias"),
        (
            "model.features.6.block.2.fc2.weight",
            "model.blocks.2.2.se.conv_expand.weight",
        ),
        ("model.features.6.block.3.0.weight", "model.blocks.2.2.conv_pwl.weight"),
        ("model.features.6.block.3.1.weight", "model.blocks.2.2.bn3.weight"),
        ("model.features.6.block.3.1.bias", "model.blocks.2.2.bn3.bias"),
        (
            "model.features.6.block.3.1.running_mean",
            "model.blocks.2.2.bn3.running_mean",
        ),
        ("model.features.6.block.3.1.running_var", "model.blocks.2.2.bn3.running_var"),
        ("model.features.7.block.0.0.weight", "model.blocks.3.0.conv_pw.weight"),
        ("model.features.7.block.0.1.weight", "model.blocks.3.0.bn1.weight"),
        ("model.features.7.block.1.1.weight", "model.blocks.3.0.bn2.weight"),
        ("model.features.7.block.0.1.bias", "model.blocks.3.0.bn1.bias"),
        ("model.features.7.block.1.1.bias", "model.blocks.3.0.bn2.bias"),
        (
            "model.features.7.block.0.1.running_mean",
            "model.blocks.3.0.bn1.running_mean",
        ),
        (
            "model.features.7.block.1.1.running_mean",
            "model.blocks.3.0.bn2.running_mean",
        ),
        ("model.features.7.block.0.1.running_var", "model.blocks.3.0.bn1.running_var"),
        ("model.features.7.block.1.1.running_var", "model.blocks.3.0.bn2.running_var"),
        (
            "model.features.7.block.0.1.num_batches_tracked",
            "model.blocks.3.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.7.block.1.1.num_batches_tracked",
            "model.blocks.3.0.bn2.num_batches_tracked",
        ),
        (
            "model.features.7.block.2.1.num_batches_tracked",
            "model.blocks.3.0.bn3.num_batches_tracked",
        ),
        ("model.features.7.block.1.0.weight", "model.blocks.3.0.conv_dw.weight"),
        ("model.features.7.block.2.0.weight", "model.blocks.3.0.conv_pwl.weight"),
        ("model.features.7.block.2.1.weight", "model.blocks.3.0.bn3.weight"),
        ("model.features.7.block.2.1.bias", "model.blocks.3.0.bn3.bias"),
        (
            "model.features.7.block.2.1.running_mean",
            "model.blocks.3.0.bn3.running_mean",
        ),
        ("model.features.7.block.2.1.running_var", "model.blocks.3.0.bn3.running_var"),
        ("model.features.8.block.0.0.weight", "model.blocks.3.1.conv_pw.weight"),
        ("model.features.8.block.0.1.weight", "model.blocks.3.1.bn1.weight"),
        ("model.features.8.block.1.1.weight", "model.blocks.3.1.bn2.weight"),
        ("model.features.8.block.0.1.bias", "model.blocks.3.1.bn1.bias"),
        ("model.features.8.block.1.1.bias", "model.blocks.3.1.bn2.bias"),
        (
            "model.features.8.block.0.1.running_mean",
            "model.blocks.3.1.bn1.running_mean",
        ),
        (
            "model.features.8.block.1.1.running_mean",
            "model.blocks.3.1.bn2.running_mean",
        ),
        ("model.features.8.block.0.1.running_var", "model.blocks.3.1.bn1.running_var"),
        ("model.features.8.block.1.1.running_var", "model.blocks.3.1.bn2.running_var"),
        (
            "model.features.8.block.0.1.num_batches_tracked",
            "model.blocks.3.1.bn1.num_batches_tracked",
        ),
        (
            "model.features.8.block.1.1.num_batches_tracked",
            "model.blocks.3.1.bn2.num_batches_tracked",
        ),
        (
            "model.features.8.block.2.1.num_batches_tracked",
            "model.blocks.3.1.bn3.num_batches_tracked",
        ),
        ("model.features.8.block.1.0.weight", "model.blocks.3.1.conv_dw.weight"),
        ("model.features.8.block.2.0.weight", "model.blocks.3.1.conv_pwl.weight"),
        ("model.features.8.block.2.1.weight", "model.blocks.3.1.bn3.weight"),
        ("model.features.8.block.2.1.bias", "model.blocks.3.1.bn3.bias"),
        (
            "model.features.8.block.2.1.running_mean",
            "model.blocks.3.1.bn3.running_mean",
        ),
        ("model.features.8.block.2.1.running_var", "model.blocks.3.1.bn3.running_var"),
        ("model.features.9.block.0.0.weight", "model.blocks.3.2.conv_pw.weight"),
        ("model.features.9.block.0.1.weight", "model.blocks.3.2.bn1.weight"),
        ("model.features.9.block.1.1.weight", "model.blocks.3.2.bn2.weight"),
        ("model.features.9.block.0.1.bias", "model.blocks.3.2.bn1.bias"),
        ("model.features.9.block.1.1.bias", "model.blocks.3.2.bn2.bias"),
        (
            "model.features.9.block.0.1.running_mean",
            "model.blocks.3.2.bn1.running_mean",
        ),
        (
            "model.features.9.block.1.1.running_mean",
            "model.blocks.3.2.bn2.running_mean",
        ),
        ("model.features.9.block.0.1.running_var", "model.blocks.3.2.bn1.running_var"),
        ("model.features.9.block.1.1.running_var", "model.blocks.3.2.bn2.running_var"),
        (
            "model.features.9.block.0.1.num_batches_tracked",
            "model.blocks.3.2.bn1.num_batches_tracked",
        ),
        (
            "model.features.9.block.1.1.num_batches_tracked",
            "model.blocks.3.2.bn2.num_batches_tracked",
        ),
        (
            "model.features.9.block.2.1.num_batches_tracked",
            "model.blocks.3.2.bn3.num_batches_tracked",
        ),
        ("model.features.9.block.1.0.weight", "model.blocks.3.2.conv_dw.weight"),
        ("model.features.9.block.2.0.weight", "model.blocks.3.2.conv_pwl.weight"),
        ("model.features.9.block.2.1.weight", "model.blocks.3.2.bn3.weight"),
        ("model.features.9.block.2.1.bias", "model.blocks.3.2.bn3.bias"),
        (
            "model.features.9.block.2.1.running_mean",
            "model.blocks.3.2.bn3.running_mean",
        ),
        ("model.features.9.block.2.1.running_var", "model.blocks.3.2.bn3.running_var"),
        ("model.features.10.block.0.0.weight", "model.blocks.3.3.conv_pw.weight"),
        ("model.features.10.block.0.1.weight", "model.blocks.3.3.bn1.weight"),
        ("model.features.10.block.1.1.weight", "model.blocks.3.3.bn2.weight"),
        ("model.features.10.block.0.1.bias", "model.blocks.3.3.bn1.bias"),
        ("model.features.10.block.1.1.bias", "model.blocks.3.3.bn2.bias"),
        (
            "model.features.10.block.0.1.running_mean",
            "model.blocks.3.3.bn1.running_mean",
        ),
        (
            "model.features.10.block.1.1.running_mean",
            "model.blocks.3.3.bn2.running_mean",
        ),
        ("model.features.10.block.0.1.running_var", "model.blocks.3.3.bn1.running_var"),
        ("model.features.10.block.1.1.running_var", "model.blocks.3.3.bn2.running_var"),
        (
            "model.features.10.block.0.1.num_batches_tracked",
            "model.blocks.3.3.bn1.num_batches_tracked",
        ),
        (
            "model.features.10.block.1.1.num_batches_tracked",
            "model.blocks.3.3.bn2.num_batches_tracked",
        ),
        (
            "model.features.10.block.2.1.num_batches_tracked",
            "model.blocks.3.3.bn3.num_batches_tracked",
        ),
        ("model.features.10.block.1.0.weight", "model.blocks.3.3.conv_dw.weight"),
        ("model.features.10.block.2.0.weight", "model.blocks.3.3.conv_pwl.weight"),
        ("model.features.10.block.2.1.weight", "model.blocks.3.3.bn3.weight"),
        ("model.features.10.block.2.1.bias", "model.blocks.3.3.bn3.bias"),
        (
            "model.features.10.block.2.1.running_mean",
            "model.blocks.3.3.bn3.running_mean",
        ),
        ("model.features.10.block.2.1.running_var", "model.blocks.3.3.bn3.running_var"),
        ("model.features.11.block.0.0.weight", "model.blocks.4.0.conv_pw.weight"),
        ("model.features.11.block.0.1.weight", "model.blocks.4.0.bn1.weight"),
        ("model.features.11.block.1.1.weight", "model.blocks.4.0.bn2.weight"),
        ("model.features.11.block.0.1.bias", "model.blocks.4.0.bn1.bias"),
        ("model.features.11.block.1.1.bias", "model.blocks.4.0.bn2.bias"),
        ("model.features.11.block.2.fc2.bias", "model.blocks.4.0.se.conv_expand.bias"),
        (
            "model.features.11.block.0.1.running_mean",
            "model.blocks.4.0.bn1.running_mean",
        ),
        (
            "model.features.11.block.1.1.running_mean",
            "model.blocks.4.0.bn2.running_mean",
        ),
        ("model.features.11.block.0.1.running_var", "model.blocks.4.0.bn1.running_var"),
        ("model.features.11.block.1.1.running_var", "model.blocks.4.0.bn2.running_var"),
        (
            "model.features.11.block.0.1.num_batches_tracked",
            "model.blocks.4.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.11.block.1.1.num_batches_tracked",
            "model.blocks.4.0.bn2.num_batches_tracked",
        ),
        (
            "model.features.11.block.3.1.num_batches_tracked",
            "model.blocks.4.0.bn3.num_batches_tracked",
        ),
        ("model.features.11.block.1.0.weight", "model.blocks.4.0.conv_dw.weight"),
        (
            "model.features.11.block.2.fc1.weight",
            "model.blocks.4.0.se.conv_reduce.weight",
        ),
        ("model.features.11.block.2.fc1.bias", "model.blocks.4.0.se.conv_reduce.bias"),
        (
            "model.features.11.block.2.fc2.weight",
            "model.blocks.4.0.se.conv_expand.weight",
        ),
        ("model.features.11.block.3.0.weight", "model.blocks.4.0.conv_pwl.weight"),
        ("model.features.11.block.3.1.weight", "model.blocks.4.0.bn3.weight"),
        ("model.features.11.block.3.1.bias", "model.blocks.4.0.bn3.bias"),
        (
            "model.features.11.block.3.1.running_mean",
            "model.blocks.4.0.bn3.running_mean",
        ),
        ("model.features.11.block.3.1.running_var", "model.blocks.4.0.bn3.running_var"),
        ("model.features.12.block.0.0.weight", "model.blocks.4.1.conv_pw.weight"),
        ("model.features.12.block.0.1.weight", "model.blocks.4.1.bn1.weight"),
        ("model.features.12.block.1.1.weight", "model.blocks.4.1.bn2.weight"),
        ("model.features.12.block.0.1.bias", "model.blocks.4.1.bn1.bias"),
        ("model.features.12.block.1.1.bias", "model.blocks.4.1.bn2.bias"),
        ("model.features.12.block.2.fc2.bias", "model.blocks.4.1.se.conv_expand.bias"),
        (
            "model.features.12.block.0.1.running_mean",
            "model.blocks.4.1.bn1.running_mean",
        ),
        (
            "model.features.12.block.1.1.running_mean",
            "model.blocks.4.1.bn2.running_mean",
        ),
        ("model.features.12.block.0.1.running_var", "model.blocks.4.1.bn1.running_var"),
        ("model.features.12.block.1.1.running_var", "model.blocks.4.1.bn2.running_var"),
        (
            "model.features.12.block.0.1.num_batches_tracked",
            "model.blocks.4.1.bn1.num_batches_tracked",
        ),
        (
            "model.features.12.block.1.1.num_batches_tracked",
            "model.blocks.4.1.bn2.num_batches_tracked",
        ),
        (
            "model.features.12.block.3.1.num_batches_tracked",
            "model.blocks.4.1.bn3.num_batches_tracked",
        ),
        ("model.features.12.block.1.0.weight", "model.blocks.4.1.conv_dw.weight"),
        (
            "model.features.12.block.2.fc1.weight",
            "model.blocks.4.1.se.conv_reduce.weight",
        ),
        ("model.features.12.block.2.fc1.bias", "model.blocks.4.1.se.conv_reduce.bias"),
        (
            "model.features.12.block.2.fc2.weight",
            "model.blocks.4.1.se.conv_expand.weight",
        ),
        ("model.features.12.block.3.0.weight", "model.blocks.4.1.conv_pwl.weight"),
        ("model.features.12.block.3.1.weight", "model.blocks.4.1.bn3.weight"),
        ("model.features.12.block.3.1.bias", "model.blocks.4.1.bn3.bias"),
        (
            "model.features.12.block.3.1.running_mean",
            "model.blocks.4.1.bn3.running_mean",
        ),
        ("model.features.12.block.3.1.running_var", "model.blocks.4.1.bn3.running_var"),
        ("model.features.13.block.0.0.weight", "model.blocks.5.0.conv_pw.weight"),
        ("model.features.13.block.0.1.weight", "model.blocks.5.0.bn1.weight"),
        ("model.features.13.block.1.1.weight", "model.blocks.5.0.bn2.weight"),
        ("model.features.13.block.0.1.bias", "model.blocks.5.0.bn1.bias"),
        ("model.features.13.block.1.1.bias", "model.blocks.5.0.bn2.bias"),
        ("model.features.13.block.2.fc2.bias", "model.blocks.5.0.se.conv_expand.bias"),
        (
            "model.features.13.block.0.1.running_mean",
            "model.blocks.5.0.bn1.running_mean",
        ),
        (
            "model.features.13.block.1.1.running_mean",
            "model.blocks.5.0.bn2.running_mean",
        ),
        ("model.features.13.block.0.1.running_var", "model.blocks.5.0.bn1.running_var"),
        ("model.features.13.block.1.1.running_var", "model.blocks.5.0.bn2.running_var"),
        (
            "model.features.13.block.0.1.num_batches_tracked",
            "model.blocks.5.0.bn1.num_batches_tracked",
        ),
        (
            "model.features.13.block.1.1.num_batches_tracked",
            "model.blocks.5.0.bn2.num_batches_tracked",
        ),
        (
            "model.features.13.block.3.1.num_batches_tracked",
            "model.blocks.5.0.bn3.num_batches_tracked",
        ),
        ("model.features.13.block.1.0.weight", "model.blocks.5.0.conv_dw.weight"),
        (
            "model.features.13.block.2.fc1.weight",
            "model.blocks.5.0.se.conv_reduce.weight",
        ),
        ("model.features.13.block.2.fc1.bias", "model.blocks.5.0.se.conv_reduce.bias"),
        (
            "model.features.13.block.2.fc2.weight",
            "model.blocks.5.0.se.conv_expand.weight",
        ),
        ("model.features.13.block.3.0.weight", "model.blocks.5.0.conv_pwl.weight"),
        ("model.features.13.block.3.1.weight", "model.blocks.5.0.bn3.weight"),
        ("model.features.13.block.3.1.bias", "model.blocks.5.0.bn3.bias"),
        (
            "model.features.13.block.3.1.running_mean",
            "model.blocks.5.0.bn3.running_mean",
        ),
        ("model.features.13.block.3.1.running_var", "model.blocks.5.0.bn3.running_var"),
        ("model.features.14.block.0.0.weight", "model.blocks.5.1.conv_pw.weight"),
        ("model.features.14.block.0.1.weight", "model.blocks.5.1.bn1.weight"),
        ("model.features.14.block.1.1.weight", "model.blocks.5.1.bn2.weight"),
        ("model.features.14.block.0.1.bias", "model.blocks.5.1.bn1.bias"),
        ("model.features.14.block.1.1.bias", "model.blocks.5.1.bn2.bias"),
        ("model.features.14.block.2.fc2.bias", "model.blocks.5.1.se.conv_expand.bias"),
        (
            "model.features.14.block.0.1.running_mean",
            "model.blocks.5.1.bn1.running_mean",
        ),
        (
            "model.features.14.block.1.1.running_mean",
            "model.blocks.5.1.bn2.running_mean",
        ),
        ("model.features.14.block.0.1.running_var", "model.blocks.5.1.bn1.running_var"),
        ("model.features.14.block.1.1.running_var", "model.blocks.5.1.bn2.running_var"),
        (
            "model.features.14.block.0.1.num_batches_tracked",
            "model.blocks.5.1.bn1.num_batches_tracked",
        ),
        (
            "model.features.14.block.1.1.num_batches_tracked",
            "model.blocks.5.1.bn2.num_batches_tracked",
        ),
        (
            "model.features.14.block.3.1.num_batches_tracked",
            "model.blocks.5.1.bn3.num_batches_tracked",
        ),
        ("model.features.14.block.1.0.weight", "model.blocks.5.1.conv_dw.weight"),
        (
            "model.features.14.block.2.fc1.weight",
            "model.blocks.5.1.se.conv_reduce.weight",
        ),
        ("model.features.14.block.2.fc1.bias", "model.blocks.5.1.se.conv_reduce.bias"),
        (
            "model.features.14.block.2.fc2.weight",
            "model.blocks.5.1.se.conv_expand.weight",
        ),
        ("model.features.14.block.3.0.weight", "model.blocks.5.1.conv_pwl.weight"),
        ("model.features.14.block.3.1.weight", "model.blocks.5.1.bn3.weight"),
        ("model.features.14.block.3.1.bias", "model.blocks.5.1.bn3.bias"),
        (
            "model.features.14.block.3.1.running_mean",
            "model.blocks.5.1.bn3.running_mean",
        ),
        ("model.features.14.block.3.1.running_var", "model.blocks.5.1.bn3.running_var"),
        ("model.features.15.block.0.0.weight", "model.blocks.5.2.conv_pw.weight"),
        ("model.features.15.block.0.1.weight", "model.blocks.5.2.bn1.weight"),
        ("model.features.15.block.1.1.weight", "model.blocks.5.2.bn2.weight"),
        ("model.features.15.block.0.1.bias", "model.blocks.5.2.bn1.bias"),
        ("model.features.15.block.1.1.bias", "model.blocks.5.2.bn2.bias"),
        ("model.features.15.block.2.fc2.bias", "model.blocks.5.2.se.conv_expand.bias"),
        (
            "model.features.15.block.0.1.running_mean",
            "model.blocks.5.2.bn1.running_mean",
        ),
        (
            "model.features.15.block.1.1.running_mean",
            "model.blocks.5.2.bn2.running_mean",
        ),
        ("model.features.15.block.0.1.running_var", "model.blocks.5.2.bn1.running_var"),
        ("model.features.15.block.1.1.running_var", "model.blocks.5.2.bn2.running_var"),
        (
            "model.features.15.block.0.1.num_batches_tracked",
            "model.blocks.5.2.bn1.num_batches_tracked",
        ),
        (
            "model.features.15.block.1.1.num_batches_tracked",
            "model.blocks.5.2.bn2.num_batches_tracked",
        ),
        (
            "model.features.15.block.3.1.num_batches_tracked",
            "model.blocks.5.2.bn3.num_batches_tracked",
        ),
        ("model.features.15.block.1.0.weight", "model.blocks.5.2.conv_dw.weight"),
        (
            "model.features.15.block.2.fc1.weight",
            "model.blocks.5.2.se.conv_reduce.weight",
        ),
        ("model.features.15.block.2.fc1.bias", "model.blocks.5.2.se.conv_reduce.bias"),
        (
            "model.features.15.block.2.fc2.weight",
            "model.blocks.5.2.se.conv_expand.weight",
        ),
        ("model.features.15.block.3.0.weight", "model.blocks.5.2.conv_pwl.weight"),
        ("model.features.15.block.3.1.weight", "model.blocks.5.2.bn3.weight"),
        ("model.features.15.block.3.1.bias", "model.blocks.5.2.bn3.bias"),
        (
            "model.features.15.block.3.1.running_mean",
            "model.blocks.5.2.bn3.running_mean",
        ),
        ("model.features.15.block.3.1.running_var", "model.blocks.5.2.bn3.running_var"),
        ("model.features.16.0.weight", "model.blocks.6.0.conv.weight"),
        ("model.features.16.1.weight", "model.blocks.6.0.bn1.weight"),
        ("model.features.16.1.bias", "model.blocks.6.0.bn1.bias"),
        ("model.features.16.1.running_mean", "model.blocks.6.0.bn1.running_mean"),
        ("model.features.16.1.running_var", "model.blocks.6.0.bn1.running_var"),
        (
            "model.features.16.1.num_batches_tracked",
            "model.blocks.6.0.bn1.num_batches_tracked",
        ),
    ]
