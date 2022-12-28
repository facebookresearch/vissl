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

import logging
import math
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper
from vissl.config import AttrDict
from vissl.models.model_helpers import DropPath, to_2tuple, trunc_normal_
from vissl.models.trunks import register_model_trunk
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.misc import set_torch_seed


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
        self.scale = qk_scale or math.pow(head_dim, -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        attn, v = self.forward_attention(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return attn, v


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
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

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        attn, v = self.attn.forward_attention(self.norm1(x))
        return attn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.feat_map_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.feat_map_size[0] * self.feat_map_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class NoPatchMasking(nn.Module):
    """No patch masking"""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, pos_embed: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return x + pos_embed


class IBOTMaskedModeling(nn.Module):
    """Block masking module inspired by iBOT (https://arxiv.org/pdf/2111.07832.pdf)
    - contains a learn-able mask embedding
    - replace the masked tokens by the learn-able embedding

    Args:
        embed_dim (int): size of the ViT embedding
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(
        self, x: torch.Tensor, pos_embed: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        mask = kwargs.get("mask", None)
        if mask is not None:
            feat_map = x[:, 1:]
            mask = mask.view(mask.shape[0], -1)
            feat_map[mask, :] = self.masked_embed.to(x.dtype)
        return x + pos_embed


class MSNDropMasking(nn.Module):
    """Drop Masking used in MSN (https://arxiv.org/pdf/2204.07141.pdf)
    - drop part of the feature map patches (never drops the class token)
    - applies the positional embeddings before the dropping

    Args:
        drop_ratio (float): ratio of tokens to drop
    """

    def __init__(self, drop_ratio: float, global_view_tokens: int):
        super().__init__()
        self.patch_drop = drop_ratio
        self.global_view_tokens = global_view_tokens

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        # Add positional embeddings now (before the dropping)
        x = x + pos_embed

        # Skip the masking for non-focal views (small sequence length)
        if x.size(1) < self.global_view_tokens + 1:
            return x
        if self.patch_drop <= 0.0:
            return x

        # Drop positions (except for the class token)
        patch_keep = 1.0 - self.patch_drop
        T_H = int(np.floor((x.shape[1] - 1) * patch_keep))
        perm = 1 + torch.randperm(x.shape[1] - 1)[:T_H]
        idx = torch.cat([torch.zeros(1, dtype=perm.dtype, device=perm.device), perm])
        return x[:, idx, :]


class FeatAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # bs, seq_len, dims = x.shape
        x = x.permute((0, 2, 1))
        return self.avg_pool(x).squeeze()


class VisionTransformer(nn.Module):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        logging.info("Building model: Vision Transformer from yaml config")

        self.model_config = model_config
        self.trunk_config = model_config.TRUNK.VISION_TRANSFORMERS
        self.img_size = self.trunk_config.IMAGE_SIZE
        self.patch_size = self.trunk_config.PATCH_SIZE
        self.in_chans = 3
        self.embed_dim = self.trunk_config.HIDDEN_DIM
        self.depth = self.trunk_config.NUM_LAYERS
        self.num_heads = self.trunk_config.NUM_HEADS
        self.mlp_ratio = self.trunk_config.MLP_DIM / self.trunk_config.HIDDEN_DIM
        self.qkv_bias = self.trunk_config.QKV_BIAS
        self.qk_scale = self.trunk_config.QK_SCALE
        self.drop_rate = self.trunk_config.DROPOUT_RATE
        self.attn_drop_rate = self.trunk_config.ATTENTION_DROPOUT_RATE
        self.drop_path_rate = self.trunk_config.DROP_PATH_RATE
        self.use_class_token = self.trunk_config.get("USE_CLASS_TOKEN", True)

        # TODO Implement hybrid backbones
        hybrid_backbone_string = None
        if "HYBRID" in self.trunk_config.keys():
            hybrid_backbone_string = self.trunk_config.HYBRID

        norm_layer = self.create_norm_layer()

        # num_features for consistency with other models
        self.num_features = self.embed_dim

        # TODO : Enable Hybrid Backbones
        if hybrid_backbone_string:
            self.patch_embed = globals()[hybrid_backbone_string](
                out_dim=self.embed_dim, img_size=self.img_size
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
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim)
            )
        else:
            self.register_parameter("cls_token", None)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.blocks = self._build_blocks(norm_layer)
        self.norm = norm_layer(self.embed_dim)

        # NOTE as per official impl, we could have a pre-logits
        # representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        # Initializes the weights of the children, except for the blocks
        # who have already been initialized in the "build_blocks".
        for name, children in self.named_children():
            if "blocks" not in name:
                self._init_weights(children)

        if self.trunk_config.MASKED_IMAGE_MODELING.NAME == "ibot":
            self.patch_masking = IBOTMaskedModeling(embed_dim=self.embed_dim)
        elif self.trunk_config.MASKED_IMAGE_MODELING.NAME == "msn":
            drop_masking_params = self.trunk_config.MASKED_IMAGE_MODELING.PARAMS
            drop_ratio = drop_masking_params.get("drop_ratio", 0.0)
            global_view_tokens = drop_masking_params.get("global_view_tokens", 196)
            self.patch_masking = MSNDropMasking(
                drop_ratio=drop_ratio, global_view_tokens=global_view_tokens
            )
        else:
            self.patch_masking = NoPatchMasking()

        self.avg_pool = FeatAvgPool()

    def create_norm_layer(self):
        return partial(nn.LayerNorm, eps=1e-6)

    def _build_blocks(self, norm_layer) -> nn.ModuleList:
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        blocks = []
        for i in range(self.depth):
            with set_torch_seed(self.model_config._MODEL_INIT_SEED + i + 1):
                block = self._build_block(dpr[i], norm_layer)
            blocks.append(block)
        return nn.ModuleList(blocks)

    def _build_block(self, dpr: float, norm_layer) -> nn.Module:
        block = Block(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
        )
        block.apply(self._init_weights)
        if self.trunk_config.CHECKPOINT_MLP:
            block.mlp = checkpoint_wrapper(block.mlp)
        if self.trunk_config.CHECKPOINT_BLOCK:
            block = checkpoint_wrapper(block)
        return block

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

    def prepare_tokens(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        feat_map_size = self.patch_embed.feat_map_size

        # Add the class token
        if self.use_class_token:
            class_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((class_tokens, x), dim=1)

        # Compute positional embedding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed, feat_map_size)

        # Use the mask generator to create a mask, apply it, and
        # combine the result with the positional embeddings
        x = self.patch_masking(x, pos_embed, **kwargs)

        # Return sequence of shape (B, 1 + H * W, C)
        return self.pos_drop(x)

    def forward_features(
        self, x: torch.Tensor, only_class_token: bool = True, **kwargs
    ):
        """
        Return the class token representation at the last layer
        """
        x = self.prepare_tokens(x, **kwargs)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if only_class_token:
            if self.use_class_token:
                return x[:, 0]
            else:
                return self.avg_pool(x)
        else:
            return x

    def get_intermediate_features(
        self, x: torch.Tensor, names: List[str]
    ) -> List[torch.Tensor]:
        """
        Given a list of feature names, return a list of the same length
        where each output correspond to the desired feature.

        The available features are:
        - blkCLS[integer] => CLS token of blk[integer]
        - concatCLS[integer] => concat of CLS token from last #"integer" blocks
        - lastCLS => CLS token of last block
        - lastMAP => feature map of the last block
        - lastBLK => CLS token + feature map of the last block
        """

        # Get feature from every intermediate block and apply norm
        interms = []
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
            interms.append(self.norm(x))

        # Then collect the desired features
        output = []
        for name in names:
            if name.startswith("blkCLS"):
                assert self.use_class_token, "Need class token to extract blkCLS"
                v = int(name.replace("blkCLS", ""))
                output.append(interms[v][:, 0])
            elif name.startswith("concatCLS"):
                assert self.use_class_token, "Need class token to extract concatCLS"
                v = int(name.replace("concatCLS", ""))
                feat = torch.cat([x[:, 0] for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name == "lastCLS":
                assert self.use_class_token, "Need class token to extract lastCLS"
                output.append(interms[-1][:, 0])
            elif name == "lastMAP":
                assert self.use_class_token, "Need class token to extract lastMAP"
                feat_map_size = self.patch_embed.feat_map_size
                feat_map = interms[-1][:, 1:]
                B, L, C = feat_map.shape
                feat_map = feat_map.reshape((B, *feat_map_size, C))
                output.append(feat_map)
            elif name == "lastBLK":
                output.append(interms[-1])
            elif name.startswith("concatBLK"):
                v = int(name.replace("concatBLK", ""))
                feat = torch.cat(interms[-v:], dim=-1)
                output.append(feat)
            elif name == "lastPOOL":
                assert (
                    not self.use_class_token
                ), "FIXIT"  # TODO - don't pool class token
                output.append(self.avg_pool(interms[-1]))
            elif name.startswith("concatPOOL"):
                assert (
                    not self.use_class_token
                ), "FIXIT"  # TODO - don't pool class token
                v = int(name.replace("concatPOOL", ""))
                feat = torch.cat([self.avg_pool(x) for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name.startswith("stridePOOL"):
                # Format is "stridePOOL_{count}_{stride}" or "stridePOOL_{count}"
                assert (
                    not self.use_class_token
                ), "FIXIT"  # TODO - don't pool class token
                name = name.replace("stridePOOL_", "")
                parts = [int(s) for s in name.split("_")]
                if len(parts) == 2:
                    count, stride = [int(s) for s in name.split("_")]
                else:
                    count = int(name)
                    stride, r = divmod(len(interms), count)
                    stride = stride + 1 if r else stride
                feat = torch.cat(
                    [self.avg_pool(x) for x in interms[::-stride][:count]], dim=-1
                )
                output.append(feat)

        return output

    def get_last_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the attention map from the last layer, with dimensions:
        (batch_size, num_heads, seq_len + 1, seq_len + 1)
        """
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk.get_attention_map(x)

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None, **kwargs
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(x, **kwargs)
            x = x.unsqueeze(0)
        elif len(out_feat_keys) == 1 and out_feat_keys[0] == "lastBLK":
            # Small optimization for cases where we want the full output
            # but not interested in intermediate features
            # TODO - extend to len(out_feat_keys) == 1
            x = self.forward_features(x, only_class_token=False, **kwargs)
            x = x.unsqueeze(0)
        else:
            # we specified a feature layer name. Follow DINO
            # (https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L159) # NOQA
            x = self.get_intermediate_features(x, out_feat_keys)
        return x

    def interpolate_pos_encoding(
        self, x: torch.Tensor, pos_embed: torch.Tensor, feat_map_size: Tuple[int, int]
    ) -> torch.Tensor:
        if not self.use_class_token:
            assert x.shape[1] == pos_embed.shape[1], "Must match dimensions"
            return pos_embed

        npatch = x.shape[1] - 1
        nembed = pos_embed.shape[1] - 1
        if npatch == nembed:
            return pos_embed

        # Remove the class embedding and compute the interpolation scale factor
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        height, width = feat_map_size
        scale_factor = math.sqrt(npatch / nembed)

        # Interpolate the position embeddings
        pos_embed = pos_embed.reshape(1, height, width, dim).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            scale_factor=scale_factor,
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Add the class embedding back and return this
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


class VisionTransformerFSDPImpl(VisionTransformer):
    def create_norm_layer(self):
        return partial(Fp32LayerNorm, eps=1e-6)

    def _build_block(self, dpr: float, norm_layer) -> nn.Module:
        block = super()._build_block(dpr, norm_layer)
        block = fsdp_wrapper(block, **self.model_config["FSDP_CONFIG"])
        return block


@register_model_trunk("vision_transformer")
def VisionTransformerDDP(model_config: AttrDict, model_name: str):
    with set_torch_seed(model_config._MODEL_INIT_SEED):
        return VisionTransformer(model_config, model_name)


@register_model_trunk("vision_transformer_fsdp")
def VisionTransformerFSDP(model_config: AttrDict, model_name: str):
    """
    Wrap the entire trunk since we need to load checkpoint before
    train_fsdp_task.py wrapping happens.
    """
    with set_torch_seed(model_config._MODEL_INIT_SEED):
        vit = VisionTransformerFSDPImpl(model_config, model_name).cuda()
    return fsdp_wrapper(vit, **model_config.FSDP_CONFIG)
