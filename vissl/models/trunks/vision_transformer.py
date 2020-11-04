# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Code modified from https://github.com/google-research/vision_transformer
and https://www.internalfb.com/D24714842, as per https://arxiv.org/abs/2010.11929
"""

import copy
from collections import OrderedDict

import copy
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict
from vissl.models.model_helpers import lecun_normal_init

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

# Todo: probabilistic image crop

class MLPBlock(nn.Sequential):
    """Transformer MLP / feed-forward block."""

    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    def __init__(
            self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate
        )  # uses correct initialization by default
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    Encoder1DBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        # Todo: Interpolate position embeddings if sequence length of x ≠
        # sequence length position embeddings
        x = x + self.pos_embedding  # should broadcast to the same shape
        return self.ln(self.layers(self.dropout(x)))


@register_model_trunk("vision_transformer")
class VisionTransformer(nn.Module):
    """
    Vision Transformer as per https://openreview.net/pdf?id=YicbFdNTTy
    """

    # Todo: logging

    def __init__(
            self,
            model_config: AttrDict,
            # image_size,
            # patch_size,
            # num_layers,
            # num_heads,
            # hidden_dim,
            # mlp_dim,
            # dropout_rate,
            # attention_dropout_rate,
            # classifier="token",
    ):
        super().__init__()
        self.trunk_config = model_config.TRUNK.TRUNK_PARAMS.VISION_TRANSFORMERS
        self.image_size = self.trunk_config.IMAGE_SIZE
        self.patch_size = self.trunk_config.PATCH_SIZE
        self.num_layers = self.trunk_config.NUM_LAYERS
        self.num_heads = self.trunk_config.NUM_HEADS
        self.hidden_dim = self.trunk_config.HIDDEN_DIM
        self.mlp_dim = self.trunk_config.MLP_DIM
        self.attention_dropout_rate = self.trunk_config.ATTENTION_DROPOUT_RATE
        self.dropout_rate = self.trunk_config.DROPOUT_RATE
        self.classifier = self.trunk_config.CLASSIFIER

        assert self.image_size % self.patch_size == 0, "Input shape " \
                                                       "indivisble by patch size"
        assert self.classifier in ["token", "gap"], "Unexpected classifier mode"

        input_channels = 3

        # conv_proj is a more efficient version of reshaping, permuting and projecting
        # the input
        self.conv_proj = nn.Conv2d(
            input_channels, self.hidden_dim, kernel_size=self.patch_size,
            stride=self.patch_size
        )

        seq_length = (self.image_size // self.patch_size) ** 2
        if self.classifier == "token":
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            self.seq_length,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout_rate,
            self.attention_dropout_rate,
        )
        self.trunk_output = nn.Identity()

        self.init_weights()

    def init_weights(self):
        lecun_normal_init(
            self.conv_proj.weight,
            fan_in=self.conv_proj.in_channels
                   * self.conv_proj.kernel_size[0]
                   * self.conv_proj.kernel_size[1],
        )
        nn.init.zeros_(self.conv_proj.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config.pop("name")
        config.pop("heads", None)
        return cls(**config)

    def forward(self, x: torch.Tensor):
        # Todo: check image size divisble by patch size
        assert x.ndim == 4, "Unexpected input shape"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # the self attention layer exects inputs in format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x = self.encoder(x)

        if self.classifier == "token":
            # just return the output for the class token
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)

        return self.trunk_output(x)
