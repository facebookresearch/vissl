# Portions Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Ross Wightman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Code modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py # NOQA
and https://github.com/facebookresearch/deit/blob/main/models.py by Matthew
Leavitt (ito@fb.com, matthew.l.leavitt@gmail.com) and Vedanuj Goswami
(vedanuj@fb.com).
"""

import copy
import logging
import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import DropPath, to_2tuple, trunc_normal_
from vissl.models.trunks import register_model_trunk


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@register_model_trunk("vision_transformer")
class VisionTransformer(nn.Module):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = copy.deepcopy(model_config.TRUNK.VISION_TRANSFORMERS)

        logging.info("Building model: Vision Transformer from yaml config")
        # Hacky workaround
        trunk_config = AttrDict({k.lower(): v for k, v in trunk_config.items()})

        img_size = trunk_config.image_size
        patch_size = trunk_config.patch_size
        in_chans = 3
        embed_dim = trunk_config.hidden_dim
        depth = trunk_config.num_layers
        num_heads = trunk_config.num_heads
        mlp_ratio = 4.0
        qkv_bias = trunk_config.qkv_bias
        qk_scale = trunk_config.qk_scale
        drop_rate = trunk_config.dropout_rate
        attn_drop_rate = trunk_config.attention_dropout_rate
        drop_path_rate = trunk_config.drop_path_rate
        hybrid_backbone_string = None
        # TODO Implement hybrid backbones
        if "HYBRID" in trunk_config.keys():
            hybrid_backbone_string = trunk_config.HYBRID
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        # TODO : Enable Hybrid Backbones
        if hybrid_backbone_string:
            self.patch_embed = globals()[hybrid_backbone_string](
                out_dim=embed_dim, img_size=img_size
            )
        # if hybrid_backbone is not None:
        #     self.patch_embed = HybridEmbed(
        #         hybrid_backbone,
        #         img_size=img_size,
        #         in_chans=in_chans,
        #         embed_dim=embed_dim,
        #     )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits
        # representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embedding", "class_token"}

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        class_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole class_tokens impl from Phil Wang, thanks
        x = torch.cat((class_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_features(self, x, names):
        interms = []

        x = self.prepare_tokens(x)

        # get feature from every intermediate block and apply norm
        for blk in self.blocks:
            x = blk(x)
            interms.append(self.norm(x))

        # feature names are as follows
        # blkCLS[integer] => CLS token of blk[integer]
        # concatCLS[integer] => concat of CLS token from last #"integer" blocks
        # lastCLS => CLS token of last block

        output = []
        for name in names:
            if name.startswith("blkCLS"):
                v = int(name.replace("blkCLS", ""))
                output.append(interms[v][:, 0])
            elif name.startswith("concatCLS"):
                v = int(name.replace("concatCLS", ""))
                feat = torch.cat([x[:, 0] for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name == "lastCLS":
                output.append(interms[-1][:, 0])
        return output

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(x)
            x = x.unsqueeze(0)
        else:
            # we specified a feature layer name. Follow DINO
            # (https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L159) # NOQA
            x = self.get_intermediate_features(x, out_feat_keys)
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
