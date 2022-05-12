# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Code modified from https://github.com/facebookresearch/xcit/blob/main/xcit.py # NOQA
"""

import logging
import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper
from vissl.config import AttrDict
from vissl.models.model_helpers import DropPath, to_2tuple, trunc_normal_
from vissl.models.trunks import register_model_trunk
from vissl.models.trunks.vision_transformer import Mlp
from vissl.utils.fsdp_utils import fsdp_wrapper
from vissl.utils.misc import set_torch_seed


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes),
    )


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.feat_map_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        num_patches = self.feat_map_size[0] * self.feat_map_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )

    def forward(self, x, padding_size=None):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3
    windows to augment the implicit communcation performed by the block diagonal
    scatter attention. Implemented using 2 layers of separable 3x3 convolutions with
    GeLU and BatchNorm2d
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        kernel_size=3,
    ):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class ClassAttention(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

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
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        qc = q[:, :, 0:1]  # CLS token
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_tkn = self.proj(cls_tkn)
        x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

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
        eta=None,
        tokens_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just
        # the CLS
        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1] = self.norm2(x[:, 0:1])

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) operation where the channels are updated using
    a weighted sum. The weights are obtained from the (softmax normalized)
    Cross-covariance matrix (Q^T K \\in d_h \\times d_h)
    """

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
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class XCABlock(nn.Module):
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
        num_tokens=196,
        eta=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = model_config.TRUNK.XCIT

        logging.info("Building model: XCiT from yaml config")
        self.model_config = model_config
        self.trunk_config = trunk_config
        self.img_size = trunk_config.IMAGE_SIZE
        self.patch_size = trunk_config.PATCH_SIZE
        self.embed_dim = trunk_config.HIDDEN_DIM
        self.depth = trunk_config.NUM_LAYERS
        self.num_heads = trunk_config.NUM_HEADS
        self.mlp_ratio = trunk_config.MLP_RATIO
        self.qkv_bias = trunk_config.QKV_BIAS
        self.qk_scale = trunk_config.QK_SCALE
        self.drop_rate = trunk_config.DROPOUT_RATE
        self.attn_drop_rate = trunk_config.ATTENTION_DROPOUT_RATE
        self.drop_path_rate = trunk_config.DROP_PATH_RATE
        self.eta = trunk_config.ETA
        self.tokens_norm = trunk_config.TOKENS_NORM

        # num_features for consistency with other models
        self.num_features = self.embed_dim

        self.patch_embed = ConvPatchEmbed(
            img_size=self.img_size, embed_dim=self.embed_dim, patch_size=self.patch_size
        )
        self.num_patches = self.patch_embed.num_patches

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.blocks = self._create_xca_blocks(norm_layer)

        self.cls_attn_blocks = self._create_cls_attn_blocks(norm_layer)
        self.norm = norm_layer(self.embed_dim)

        self.pos_embeder = PositionalEncodingFourier(dim=self.embed_dim)
        self.use_pos = True

        # Initialize weights
        trunc_normal_(self.cls_token, std=0.02)
        self.patch_embed.apply(self._init_weights)
        self.pos_embeder.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def _create_xca_blocks(self, norm_layer):
        return nn.ModuleList(
            [self._create_xca_block(norm_layer, depth=i) for i in range(self.depth)]
        )

    def _create_xca_block(self, norm_layer, depth: int):
        with set_torch_seed(self.model_config._MODEL_INIT_SEED + depth + 1):
            block = XCABlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.drop_path_rate,
                norm_layer=norm_layer,
                num_tokens=self.num_patches,
                eta=self.eta,
            )
            block.apply(self._init_weights)
            return block

    def _create_cls_attn_blocks(self, norm_layer, depth: int = 2):
        return nn.ModuleList(
            [self._create_cls_attn_block(norm_layer, depth=i) for i in range(depth)]
        )

    def _create_cls_attn_block(self, norm_layer, depth: int):
        with set_torch_seed(
            self.model_config._MODEL_INIT_SEED + self.depth + depth + 1
        ):
            block = ClassAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                norm_layer=norm_layer,
                eta=self.eta,
                tokens_norm=self.tokens_norm,
            )
            block.apply(self._init_weights)
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
        return {"pos_embed", "cls_token", "dist_token"}

    def _get_pos_encoding(self, B, Hp, Wp, x):
        return self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)

    def _add_class_token(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

    def forward_features(self, x):
        B = x.shape[0]

        # Patch embedding
        x, (Hp, Wp) = self.patch_embed(x)
        if self.use_pos:
            pos_encoding = self._get_pos_encoding(B, Hp, Wp, x)
            x = x + pos_encoding
        x = self.pos_drop(x)

        # Go through the XCA blocks
        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        # Add the class token and go through the class attention blocks
        x = self._add_class_token(x)
        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        # Select the class token and returns it
        x = self.norm(x)[:, 0]
        return x

    def get_intermediate_features(
        self, x: torch.Tensor, names: List[str]
    ) -> List[torch.Tensor]:
        """
        Given a list of feature names, return a list of the same length
        where each output correspond to the desired feature.

        The available features are:
        - lastCLS => CLS token of last block
        - lastMAP => feature map of the last block
        """
        B = x.shape[0]

        # Patch embedding
        x, (Hp, Wp) = self.patch_embed(x)
        if self.use_pos:
            pos_encoding = self._get_pos_encoding(B, Hp, Wp, x)
            x = x + pos_encoding
        x = self.pos_drop(x)

        # Go through the XCA blocks
        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        # Add the class token and go through the class attention blocks
        x = self._add_class_token(x)
        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        # Select the features we are interested in and returns them
        output = []
        for name in names:
            if name == "lastCLS":
                output.append(self.norm(x)[:, 0])
            elif name == "lastMAP":
                feat_map_size = self.patch_embed.feat_map_size
                feat_map = self.norm(x)[:, 1:]
                B, L, C = feat_map.shape
                feat_map = feat_map.reshape((B, *feat_map_size, C))
                output.append(feat_map)
        return output

    def forward(self, x, out_feat_keys: Optional[List[str]] = None):
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(x)
            return [x]
        else:
            return self.get_intermediate_features(x, out_feat_keys)


class XCiT_FSDP(XCiT):
    """
    XCiT where blocks are wrapped by FSDP
    """

    def _create_xca_blocks(self, norm_layer):
        blocks = []
        for i in range(self.depth):
            block = self._create_xca_block(norm_layer, depth=i)
            if self.trunk_config.CHECKPOINT_XCA_BLOCK:
                block = checkpoint_wrapper(block)
            block = fsdp_wrapper(block, **self.model_config.FSDP_CONFIG)
            blocks.append(block)
        return nn.ModuleList(blocks)

    def _create_cls_attn_blocks(self, norm_layer, depth: int = 2):
        blocks = []
        for i in range(self.depth):
            block = self._create_cls_attn_block(norm_layer, depth=i)
            if self.trunk_config.CHECKPOINT_ATTN_BLOCK:
                block = checkpoint_wrapper(block)
            block = fsdp_wrapper(block, **self.model_config.FSDP_CONFIG)
            blocks.append(block)
        return nn.ModuleList(blocks)


@register_model_trunk("xcit")
def create_XCiT_DDP(model_config: AttrDict, model_name: str):
    with set_torch_seed(model_config._MODEL_INIT_SEED):
        return XCiT(model_config, model_name)


@register_model_trunk("xcit_fsdp")
def create_XCiT_FSDP(model_config: AttrDict, model_name: str):
    with set_torch_seed(model_config._MODEL_INIT_SEED):
        xcit = XCiT_FSDP(model_config, model_name)
        return fsdp_wrapper(xcit, **model_config.FSDP_CONFIG)
