"""SkySense Vision Transformer backbone (pure PyTorch + HuggingFace).

Handles Sentinel-2 multispectral and Sentinel-1 SAR imagery.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_skysense import SkySenseViTConfig
from .modeling_utils import DropPath, FFN, PatchEmbed, to_2tuple


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer for the Vision Transformer.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        feedforward_channels (int): FFN hidden dimension.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        drop_path_rate (float): Drop path rate. Default: 0.0.
        num_fcs (int): Number of FC layers in FFN. Default: 2.
        qkv_bias (bool): QKV bias. Default: True.
        with_cp (bool): Gradient checkpointing. Default: False.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_fcs: int = 2,
        qkv_bias: bool = True,
        with_cp: bool = False,
    ):
        super().__init__()
        self.with_cp = with_cp

        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            bias=qkv_bias,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(drop_rate)

        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            drop_path=drop_path_rate,
            act_layer=nn.GELU,
            add_identity=True,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner_forward(x):
            # Pre-norm attention with residual
            residual = x
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            attn_out = self.proj_drop(attn_out)
            x = residual + self.drop_path(attn_out)

            # Pre-norm FFN with residual (FFN handles its own residual)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        return x


class SkySenseViTPreTrainedModel(PreTrainedModel):
    """Base class for SkySense Vision Transformer models."""

    config_class = SkySenseViTConfig
    base_model_prefix = "skysense_vit"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights following jax_impl."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class SkySenseViTModel(SkySenseViTPreTrainedModel):
    """SkySense Vision Transformer backbone.

    A pure PyTorch + HuggingFace implementation of the ViT used in SkySense
    for Sentinel-2 multispectral and Sentinel-1 SAR imagery.
    """

    def __init__(self, config: SkySenseViTConfig):
        super().__init__(config)

        img_size = to_2tuple(config.img_size)
        self.img_size = img_size
        self.patch_size = config.patch_size
        self.with_cls_token = config.with_cls_token
        self.output_cls_token = config.output_cls_token
        self.interpolate_mode = 'bicubic'

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=config.in_channels,
            embed_dims=config.embed_dims,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            norm_layer=nn.LayerNorm if config.patch_norm else None,
        )

        num_patches = (img_size[0] // config.patch_size) * (
            img_size[1] // config.patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dims)
        )
        self.drop_after_pos = nn.Dropout(p=config.drop_rate)

        # Resolve out_indices
        out_indices = list(config.out_indices)
        resolved = []
        for idx in out_indices:
            if idx < 0:
                idx = config.num_layers + idx
            resolved.append(idx)
        self.out_indices = resolved

        # Stochastic depth (computed without tensors for meta-device compat)
        num_layers = config.num_layers
        if num_layers > 1:
            dpr = [config.drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]
        else:
            dpr = [0.0]

        # Transformer encoder layers
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=config.embed_dims,
                    num_heads=config.num_heads,
                    feedforward_channels=config.mlp_ratio * config.embed_dims,
                    attn_drop_rate=config.attn_drop_rate,
                    drop_rate=config.drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=2,
                    qkv_bias=config.qkv_bias,
                    with_cp=config.with_cp,
                )
            )

        # Final norm
        self.final_norm = config.final_norm
        if config.final_norm:
            self.norm = nn.LayerNorm(config.embed_dims)

        self.post_init()

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode='bicubic'):
        """Resize position embeddings via interpolation.

        Args:
            pos_embed (torch.Tensor): Position embedding of shape (B, L, C),
                where L = 1 (cls_token) + pos_h * pos_w.
            input_shape (tuple[int, int]): Target spatial size (H, W).
            pos_shape (tuple[int, int]): Original spatial size (pos_h, pos_w).
            mode (str): Interpolation mode. Default: 'bicubic'.

        Returns:
            torch.Tensor: Resized position embedding of shape (B, 1 + H*W, C).
        """
        assert pos_embed.ndim == 3
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=input_shape,
            align_corners=False,
            mode=mode,
        )
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def _pos_embedding(self, patched_img, hw_shape, pos_embed):
        """Apply position embedding with optional interpolation."""
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size
            pos_embed = self.resize_pos_embed(
                pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode
            )
        return self.drop_after_pos(patched_img + pos_embed)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            pixel_values: (B, C, H, W) input tensor.
            output_hidden_states: Return all hidden states.
            return_dict: Return BaseModelOutput.

        Returns:
            Feature maps or BaseModelOutput.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B = pixel_values.shape[0]

        x, hw_shape = self.patch_embed(pixel_values)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embedding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            x = x[:, 1:]

        all_hidden_states = () if output_hidden_states else None
        feature_maps = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm(x)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            if i in self.out_indices:
                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                B_, _, C = out.shape
                out = out.reshape(
                    B_, hw_shape[0], hw_shape[1], C
                ).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                feature_maps.append(out)

        if not return_dict:
            return tuple(feature_maps)

        return BaseModelOutput(
            last_hidden_state=feature_maps[-1] if feature_maps else x,
            hidden_states=all_hidden_states,
        )
