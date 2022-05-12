# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import copy
import logging
import math

import torch
import torch.nn as nn
from vissl.models.model_helpers import DropPath as PathDropout, to_2tuple, trunc_normal_
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict


"""
The ConViT is a vision transformer with modified self-attention layers that
can be initialized such that they are equivalent to a convolutional layer,
but can learn to be non-local over the course of training. This "inititialize
to be local, learn to be non-local" scheme has been shown to be extremely
data efficient, and outperform DeiTs.
ConViT paper: TODO paper link
Implementation modified from
https://github.com/sdascoli/deit/blob/local_init/local_init.py
"""


class Mlp(nn.Module):
    """
    Pretty standard MLP.
    :param in_dim: integer input dimensionality
    :param mlp_dim: integer mlp dimensionality. Defaults to in_dim.
    :param out_dim: integer output dimensionality. Defaults to in_dim.
    :param act_layer: activation function. Defaults to nn.GELU
    :param drop: dropout probability. Defaults to 0
    """

    def __init__(
        self, in_dim, mlp_dim=None, out_dim=None, act_layer=nn.GELU, dropout_rate=0.0
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        mlp_dim = mlp_dim or in_dim
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = act_layer()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, out_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class GPSA(nn.Module):
    """
    Gated positional self-attention. This is what makes a ConViT a ConViT.
    The "gating" refers to lambda, the parameter that controls the relative
    contribution (i.e. gating) of positional vs. content-based attention in
    each head.

    """

    def __init__(
        self,
        embed_dim,  # Hidden dimensionality
        num_heads=9,  # Number of attention heads. Should be a square number
        qkv_bias=False,  # Query, key, and value bias
        qk_scale=None,  # Scale query and key
        attention_dropout_rate=0.0,  # Dropout on attention
        proj_dropout_rate=0.0,  # Dropout on linear projection after
        # attention
        locality_strength=1.0,  # Determines how much the positional
        # attention is focused on the patch of maximal attention. "Alpha"
        # in the paper. Equivalent to the temperature of positional
        # attention softmax. When alpha is large, attention is focused
        # only on local patches; when alpha is small, the attention is
        # spread out into a larger area
        locality_dim=10,  # Dimensionality of position embeddings. "D_pos"
        # in the paper
        use_local_init=True,  # Local initialization. This is what makes
        # a convit convolutional.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qk = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attention_dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.pos_proj = nn.Linear(3 * locality_dim, num_heads)
        self.proj_drop = nn.Dropout(proj_dropout_rate)
        self.locality_strength = locality_strength
        self.locality_dim = locality_dim
        self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.apply(self._init_weights)
        if use_local_init:
            self.init_local(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        attn = self.get_attention(x)
        v = (
            self.v(x)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        v = v[0]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        # Get relative position indices if they don't exist or are the
        # incorrect size. This allows accommodating different resolution inputs.
        if not hasattr(self, "rel_indices") or self.rel_indices.size(1) != N:
            self.set_rel_indices(N)
        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk[0], qk[1]
        # Positional attention
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        pos_score /= self.locality_dim
        pos_score = pos_score.softmax(dim=-1)
        # Content-based attention
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        # This is where the gating happens
        attn = (1.0 - torch.sigmoid(self.alpha)) * patch_score + torch.sigmoid(
            self.alpha
        ) * pos_score
        # Normalize
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        """
        Used for visualizing and quantifying attention maps.
        Args:
            x: Input data
            return_map: Whether to return attention map or not

        Returns:
            dist: mean(attention distance weighted by attention magnitude)
            attn_map: heat map of attention values across input

        """
        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum(
            "nm,hnm->h", (distances, attn_map)
        ).mean()  # average over heads
        dist = dist.item() / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def init_local(self, locality_strength=1.0):
        """
        Initialize positional self-attention layer to be local (i.e.
        equivalent to convolution).
        Args:
            locality_strength: Controls strength of locality. See explanation
            above.
        """

        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1

        kernel_size = int(self.num_heads**0.5)
        if kernel_size % 2 == 0:
            center = (kernel_size - 1) / 2
        else:
            center = kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, : 3 * self.locality_dim] = -1
                self.pos_proj.weight.data[position, : 2 * self.locality_dim] = (
                    2 * (h1 - center) * locality_distance
                )
                self.pos_proj.weight.data[position, : 1 * self.locality_dim] = (
                    2 * (h2 - center) * locality_distance
                )
        self.pos_proj.weight.data *= locality_strength

    def set_rel_indices(self, num_patches):
        # Set relative position embeddings
        img_size = int(num_patches**0.5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3 * self.locality_dim)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:, :, :, : 3 * self.locality_dim] = indd.unsqueeze(-1)
        rel_indices[:, :, :, : 2 * self.locality_dim] = indy.unsqueeze(-1)
        rel_indices[:, :, :, : 1 * self.locality_dim] = indx.unsqueeze(-1)
        self.register_buffer("rel_indices", rel_indices, persistent=False)
        # Hacky way of setting device
        device = list(self.parameters())[0].device
        self.rel_indices = self.rel_indices.to(device)


class SelfAttention(nn.Module):
    """
    Vanilla self-attention
    """

    def __init__(
        self,
        embed_dim,  # Hidden dimensionality
        num_heads=8,  # Number of attention heads
        qkv_bias=False,  # Query, key, and value bias
        qk_scale=None,  # Scale query and key
        attention_dropout_rate=0.0,  # Dropout on attention
        proj_dropout_rate=0.0,  # Post-attention dropout on linear
        # projection
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qkv[0], qkv[1]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**0.5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx**2 + indy**2
        distances = indd**0.5
        distances = distances.to("cuda")

        dist = torch.einsum("nm,hnm->h", (distances, attn_map)).mean()
        dist = dist.item() / N
        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):
    """
    Self-attention + MLP block. One full "layer".
    """

    def __init__(
        self,
        attention_module,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        mlp_dim=None,
        qkv_bias=False,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = attention_module(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            **kwargs,
        )
        if drop_path_rate > 0.0:
            self.drop_path = PathDropout(drop_path_rate)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        if mlp_dim is None:
            mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_dim=embed_dim,
            mlp_dim=mlp_dim,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
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


# TODO: Implement
class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the
                # exact dim of the output feature map for all networks,
                # the feature metadata has reliable channel and stride info,
                # but using stride to calc feature dim requires info about
                # padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@register_model_trunk("convit")
class ConViT(nn.Module):
    """Vision transformer with locally-initializable self-attention layers"""

    def __init__(self, model_config, model_name):
        super().__init__()
        trunk_config = copy.deepcopy(model_config.TRUNK.CONVIT)
        trunk_config.update(model_config.TRUNK.VISION_TRANSFORMERS)

        logging.info("Building model: ConViT from yaml config")
        # Hacky workaround
        trunk_config = AttrDict({k.lower(): v for k, v in trunk_config.items()})

        image_size = trunk_config.image_size
        patch_size = trunk_config.patch_size
        classifier = trunk_config.classifier
        assert image_size % patch_size == 0, "Input shape indivisible by patch size"
        assert classifier in ["token", "gap"], "Unexpected classifier mode"
        n_gpsa_layers = trunk_config.n_gpsa_layers
        class_token_in_local_layers = trunk_config.class_token_in_local_layers
        mlp_dim = trunk_config.mlp_dim
        embed_dim = trunk_config.hidden_dim
        locality_dim = trunk_config.locality_dim
        attention_dropout_rate = trunk_config.attention_dropout_rate
        dropout_rate = trunk_config.dropout_rate
        drop_path_rate = trunk_config.drop_path_rate
        num_layers = trunk_config.num_layers
        locality_strength = trunk_config.locality_strength
        num_heads = trunk_config.num_heads
        qkv_bias = trunk_config.qkv_bias
        qk_scale = trunk_config.qk_scale
        use_local_init = trunk_config.use_local_init

        hybrid_backbone = None
        if "hybrid" in trunk_config.keys():
            hybrid_backbone = trunk_config.hybrid

        in_chans = 3
        # TODO: Make this configurable
        norm_layer = nn.LayerNorm

        self.classifier = classifier
        self.n_gpsa_layers = n_gpsa_layers
        self.class_token_in_local_layers = class_token_in_local_layers
        # For consistency with other models
        self.num_features = self.embed_dim = self.hidden_dim = embed_dim
        self.locality_dim = locality_dim

        # Hybrid backbones not tested
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=image_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        seq_length = (image_size // patch_size) ** 2
        self.seq_length = seq_length

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        if class_token_in_local_layers:
            seq_length += 1

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        layers = []
        for i in range(num_layers):
            if i < self.n_gpsa_layers:
                if locality_strength > 0:
                    layer_locality_strength = locality_strength
                else:
                    layer_locality_strength = 1 / (i + 1)
                layers.append(
                    AttentionBlock(
                        attention_module=GPSA,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        dropout_rate=dropout_rate,
                        attention_dropout_rate=attention_dropout_rate,
                        drop_path_rate=dpr[i],
                        norm_layer=norm_layer,
                        locality_strength=layer_locality_strength,
                        locality_dim=self.locality_dim,
                        use_local_init=use_local_init,
                    )
                )
            else:
                layers.append(
                    AttentionBlock(
                        attention_module=SelfAttention,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        dropout_rate=dropout_rate,
                        attention_dropout_rate=attention_dropout_rate,
                        drop_path_rate=dpr[i],
                        norm_layer=norm_layer,
                    )
                )
        self.blocks = nn.ModuleList(layers)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embedding, std=0.02)
        trunc_normal_(self.class_token, std=0.02)
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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole class_tokens impl from Phil Wang, thanks
        class_tokens = self.class_token.expand(B, -1, -1)

        pos_embedding = self.interpolate_pos_embedding_to_data(x, self.pos_embedding)
        x = x + pos_embedding
        x = self.pos_drop(x)

        if self.class_token_in_local_layers:
            x = torch.cat((class_tokens, x), dim=1)
        for u, blk in enumerate(self.blocks):
            if u == self.n_gpsa_layers and not self.class_token_in_local_layers:
                x = torch.cat((class_tokens, x), dim=1)
            x = blk(x)

        x = self.norm(x)

        if self.classifier == "token":
            # just return the output for the class token
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x

    def forward(self, x, out_feat_keys):
        x = self.forward_features(x)
        x = x.unsqueeze(0)
        return x

    def interpolate_position_embedding(self, pos_embedding):
        """
        Interpolate a passed position embedding to fit the position embedding of
        this model. Typically called when fine-tuning on higher resolution
        images than the model was pre-trained on. Called by
        viss/utils/checkpoint.py when loading state_dict.
        """
        # shape of pos_embedding is (1, seq_length, hidden_dim)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(
                f"Unexpected position embedding shape: {pos_embedding.shape}"
            )
        if hidden_dim != self.embed_dim:
            raise ValueError(
                f"Position embedding hidden_dim incorrect: {self.embed_dim}"
                f", expected: {self.embed_dim}"
            )
        new_seq_length = self.seq_length

        if new_seq_length != seq_length:
            # need to interpolate the weights for the position embedding
            # we do this by reshaping the positions embeddings to a 2d grid,
            # performing an interpolation in the (h, w) space and then
            # reshaping back to a 1d grid

            # (seq_length, 1, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding = pos_embedding.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            assert (
                seq_length_1d * seq_length_1d == seq_length
            ), "seq_length is not a perfect square"

            logging.info(
                "Interpolating the position embeddings from image "
                f"{seq_length_1d * self.patch_size} to size"
                f" {self.image_size}"
            )

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding = pos_embedding.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = self.image_size // self.patch_size

            # use bicubic interpolation - it gives significantly better
            # results in the test `test_resolution_change`
            new_pos_embedding = torch.nn.functional.interpolate(
                pos_embedding,
                size=new_seq_length_1d,
                mode="bicubic",
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) ->
            # (1, hidden_dim, new_seq_l)
            new_pos_embedding = new_pos_embedding.reshape(1, hidden_dim, new_seq_length)
            # (1, hidden_dim, new_seq_length) -> (new_seq_length, 1, hidden_dim)
            new_pos_embedding = new_pos_embedding.permute(0, 2, 1)
            return new_pos_embedding
        else:
            return pos_embedding

    def interpolate_pos_embedding_to_data(self, x, pos_embed):
        """
        Interpolates position embeddings for on a single instance basis (e.g.
        for a single forward pass) wrt passed data, in contrast to
        interpolate_position_embedding(), which interpolates
        the stored parameter. This function is needed if passing images of a
        size different from that which the model was initialized to accept.
        """
        data_seq_length = x.shape[1]
        pos_embed_seq_length = pos_embed.shape[1]
        if data_seq_length == pos_embed_seq_length:
            return pos_embed
        dim = x.shape[-1]
        pos_embed = pos_embed.reshape(
            1,
            int(math.sqrt(pos_embed_seq_length)),
            int(math.sqrt(pos_embed_seq_length)),
            dim,
        )
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            scale_factor=math.sqrt(data_seq_length / pos_embed_seq_length),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed
