"""SkySense Swin Transformer V2 backbone (pure PyTorch + HuggingFace).

Handles high-resolution optical imagery (RGB/RGBNIR).
"""

from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_skysense import SkySenseSwinV2Config
from .modeling_utils import (
    DropPath,
    FFN,
    PatchEmbed,
    PatchMerging,
    ShiftWindowMSA,
    to_2tuple,
)


class SwinBlockV2(nn.Module):
    """Swin Transformer V2 block with post-normalization.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size. Default: 8.
        shift (bool): Shift the attention window. Default: False.
        extra_norm (bool): Extra norm at end of block. Default: False.
        ffn_ratio (float): FFN expansion ratio. Default: 4.0.
        drop_path (float): Drop path rate. Default: 0.0.
        pad_small_map (bool): Pad small maps. Default: False.
        with_cp (bool): Gradient checkpointing. Default: False.
        pretrained_window_size (int): Pretrained window size. Default: 0.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 8,
        shift: bool = False,
        extra_norm: bool = False,
        ffn_ratio: float = 4.0,
        drop_path: float = 0.0,
        pad_small_map: bool = False,
        with_cp: bool = False,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.extra_norm = extra_norm

        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            drop_path=drop_path,
            pad_small_map=pad_small_map,
            pretrained_window_size=pretrained_window_size,
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=int(embed_dims * ffn_ratio),
            num_fcs=2,
            drop_path=drop_path,
            act_layer=nn.GELU,
            add_identity=False,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        if self.extra_norm:
            self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        def _inner_forward(x):
            # Post normalization
            identity = x
            x = self.attn(x, hw_shape)
            x = self.norm1(x)
            x = x + identity

            identity = x
            x = self.ffn(x)
            x = self.norm2(x)
            x = x + identity

            if self.extra_norm:
                x = self.norm3(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        return x


class SwinBlockV2Sequence(nn.Module):
    """Sequence of Swin Transformer V2 blocks with optional downsample.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size. Default: 8.
        downsample (bool): Apply downsample. Default: False.
        drop_paths (list or float): Drop path rates. Default: 0.0.
        with_cp (bool): Gradient checkpointing. Default: False.
        pad_small_map (bool): Pad small maps. Default: False.
        extra_norm_every_n_blocks (int): Extra norm interval. Default: 0.
        pretrained_window_size (int): Pretrained window size. Default: 0.
        is_post_norm_downsample (bool): Post-norm in downsample. Default: True.
    """

    def __init__(
        self,
        embed_dims: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        downsample: bool = False,
        drop_paths: Union[Sequence[float], float] = 0.0,
        with_cp: bool = False,
        pad_small_map: bool = False,
        extra_norm_every_n_blocks: int = 0,
        pretrained_window_size: int = 0,
        is_post_norm_downsample: bool = True,
    ):
        super().__init__()

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if downsample:
            self.out_channels = 2 * embed_dims
            self.downsample = PatchMerging(
                in_channels=embed_dims,
                out_channels=self.out_channels,
                is_post_norm=is_post_norm_downsample,
            )
        else:
            self.out_channels = embed_dims
            self.downsample = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            extra_norm = (
                extra_norm_every_n_blocks > 0
                and (i + 1) % extra_norm_every_n_blocks == 0
            )
            block = SwinBlockV2(
                embed_dims=self.out_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift=(i % 2 == 1),
                extra_norm=extra_norm,
                drop_path=drop_paths[i],
                with_cp=with_cp,
                pad_small_map=pad_small_map,
                pretrained_window_size=pretrained_window_size,
            )
            self.blocks.append(block)

    def forward(
        self, x: torch.Tensor, in_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.downsample is not None:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape

        for block in self.blocks:
            x = block(x, out_shape)

        return x, out_shape


class SkySenseSwinV2PreTrainedModel(PreTrainedModel):
    """Base class for SkySense Swin Transformer V2 models."""

    config_class = SkySenseSwinV2Config
    base_model_prefix = "skysense_swinv2"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class SkySenseSwinV2Model(SkySenseSwinV2PreTrainedModel):
    """SkySense Swin Transformer V2 backbone.

    A pure PyTorch + HuggingFace implementation of the Swin Transformer V2
    used in SkySense for high-resolution optical remote sensing imagery.
    """

    def __init__(self, config: SkySenseSwinV2Config):
        super().__init__(config)

        self.num_layers = len(config.depths)
        self.out_indices = config.out_indices

        # Window sizes per stage
        if isinstance(config.window_size, int):
            window_sizes = [config.window_size] * self.num_layers
        else:
            window_sizes = list(config.window_size)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=config.in_channels,
            embed_dims=config.embed_dims,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            norm_layer=nn.LayerNorm,
            input_size=config.img_size,
        )

        # Optional absolute position embedding
        self.use_abs_pos_embed = config.use_abs_pos_embed
        if self.use_abs_pos_embed:
            patch_resolution = self.patch_embed.init_out_size
            num_patches = patch_resolution[0] * patch_resolution[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, config.embed_dims)
            )

        self.drop_after_pos = nn.Dropout(p=config.drop_rate)

        # Stochastic depth decay (computed without tensors for meta-device compat)
        total_depth = sum(config.depths)
        if total_depth > 1:
            dpr = [config.drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]
        else:
            dpr = [0.0]

        # Build stages
        self.stages = nn.ModuleList()
        embed_dims_list = [config.embed_dims]
        for i, (depth, num_heads) in enumerate(
            zip(config.depths, config.num_heads)
        ):
            stage = SwinBlockV2Sequence(
                embed_dims=embed_dims_list[-1],
                depth=depth,
                num_heads=num_heads,
                window_size=window_sizes[i],
                downsample=(i > 0),
                drop_paths=dpr[:depth],
                with_cp=config.with_cp,
                pad_small_map=config.pad_small_map,
                extra_norm_every_n_blocks=config.extra_norm_every_n_blocks,
                pretrained_window_size=config.pretrained_window_sizes[i],
                is_post_norm_downsample=config.is_post_norm_downsample,
            )
            self.stages.append(stage)
            dpr = dpr[depth:]
            embed_dims_list.append(stage.out_channels)

        # Output norms
        for i in self.out_indices:
            norm = nn.LayerNorm(embed_dims_list[i + 1])
            self.add_module(f"norm{i}", norm)

        self.post_init()

    def _delete_reinit_params(self, state_dict, prefix, *args, **kwargs):
        """Delete relative_position_index and relative_coords_table from state_dict."""
        keys_to_delete = [
            k for k in state_dict.keys()
            if 'relative_position_index' in k or 'relative_coords_table' in k
        ]
        for k in keys_to_delete:
            del state_dict[k]

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            pixel_values: (B, C, H, W) input image tensor.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a BaseModelOutput.

        Returns:
            BaseModelOutput or tuple of feature maps.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x, hw_shape = self.patch_embed(pixel_values)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        all_hidden_states = () if output_hidden_states else None
        feature_maps = []

        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(x)
                out = out.view(
                    -1, *hw_shape, stage.out_channels
                ).permute(0, 3, 1, 2).contiguous()
                feature_maps.append(out)

        if not return_dict:
            return tuple(feature_maps)

        return BaseModelOutput(
            last_hidden_state=feature_maps[-1] if feature_maps else x,
            hidden_states=all_hidden_states,
        )
