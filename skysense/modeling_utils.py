"""SkySense: Pure PyTorch + HuggingFace Transformers implementation.

Shared utility modules used across SkySense model implementations.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    """Convert to a 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    Args:
        drop_prob (float): Probability of dropping a path. Default: 0.0.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 96.
        kernel_size (int): Kernel size of the projection. Default: 4.
        stride (int): Stride of the projection. Default: 4.
        padding (int): Padding of the projection. Default: 0.
        norm_layer (nn.Module or None): Normalization layer. Default: nn.LayerNorm.
        input_size (int or tuple or None): Input resolution for calculating output size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        norm_layer: Optional[type] = nn.LayerNorm,
        input_size: Optional[int] = None,
    ):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self.norm = norm_layer(embed_dims) if norm_layer else nn.Identity()

        # Compute init output size if input_size is given
        if input_size is not None:
            input_size = to_2tuple(input_size)
            self.init_out_size = (
                (input_size[0] - kernel_size + 2 * padding) // stride + 1,
                (input_size[1] - kernel_size + 2 * padding) // stride + 1,
            )
        else:
            self.init_out_size = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.projection(x)  # (B, C, H, W)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, out_size


class FFN(nn.Module):
    """Feed-Forward Network.

    Args:
        embed_dims (int): Input dimension.
        feedforward_channels (int): Hidden dimension.
        num_fcs (int): Number of FC layers. Default: 2.
        ffn_drop (float): Dropout rate. Default: 0.0.
        drop_path (float): Drop path rate. Default: 0.0.
        act_layer (nn.Module): Activation layer class. Default: nn.GELU.
        add_identity (bool): Whether to add identity connection. Default: True.
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        add_identity: bool = True,
    ):
        super().__init__()
        assert num_fcs >= 2, f"num_fcs must be >= 2, got {num_fcs}"
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.add_identity = add_identity

        layers = []
        in_channels = embed_dims
        for i in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(act_layer())
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.layers(x)
        out = self.drop_path(out)
        if self.add_identity:
            if identity is None:
                identity = x
            out = out + identity
        return out


class WindowMSAV2(nn.Module):
    """Window-based Multi-head Self-Attention for Swin Transformer V2.

    Uses cosine attention and log-spaced continuous position bias (log-CPB).

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (Wh, Ww).
        pretrained_window_size (tuple[int]): Pretrained window size for CPB. Default: (0, 0).
        qkv_bias (bool): If True, add learnable bias to q, k, v. Default: True.
        attn_drop (float): Attention dropout rate. Default: 0.0.
        proj_drop (float): Output projection dropout rate. Default: 0.0.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        pretrained_window_size: Tuple[int, int] = (0, 0),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))))

        # MLP for continuous relative position bias (log-CPB)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Build relative coords table
        self._build_relative_coords_table()
        # Build relative position index
        self._build_relative_position_index()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _build_relative_coords_table(self):
        """Build the relative coordinates table for log-CPB."""
        Wh, Ww = self.window_size
        # Table of relative coordinates
        coords_h = torch.arange(-(Wh - 1), Wh, dtype=torch.float32)
        coords_w = torch.arange(-(Ww - 1), Ww, dtype=torch.float32)
        coords_table = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij')
        ).flatten(1).transpose(0, 1).unsqueeze(0)  # (1, (2Wh-1)*(2Ww-1), 2)

        # Normalize to [-1, 1] and apply log-scale
        if self.pretrained_window_size[0] > 0:
            coords_table[:, :, 0] /= (self.pretrained_window_size[0] - 1)
            coords_table[:, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            coords_table[:, :, 0] /= (Wh - 1)
            coords_table[:, :, 1] /= (Ww - 1)
        coords_table *= 8  # normalize to -8, 8
        coords_table = (
            torch.sign(coords_table)
            * torch.log2(torch.abs(coords_table) + 1.0)
            / math.log2(8)
        )
        self.register_buffer("relative_coords_table", coords_table)

    def _build_relative_position_index(self):
        """Build the pairwise relative position index for each window token."""
        Wh, Ww = self.window_size
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.view(2, -1)

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

    def _compute_position_bias(self, N):
        """Compute relative position bias, supporting dynamic window sizes.

        The log-CPB (Continuous Position Bias) MLP can generalize to any window
        size by computing bias from normalized relative coordinates.
        """
        init_N = self.window_size[0] * self.window_size[1]
        if N == init_N:
            # Use pre-built tables
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N, N, -1)
        else:
            # Dynamic: compute for actual window size on-the-fly
            Wh = Ww = int(math.sqrt(N))
            coords_h = torch.arange(-(Wh - 1), Wh, dtype=torch.float32, device=self.logit_scale.device)
            coords_w = torch.arange(-(Ww - 1), Ww, dtype=torch.float32, device=self.logit_scale.device)
            coords_table = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing='ij')
            ).flatten(1).transpose(0, 1).unsqueeze(0)
            if self.pretrained_window_size[0] > 0:
                coords_table[:, :, 0] /= (self.pretrained_window_size[0] - 1)
                coords_table[:, :, 1] /= (self.pretrained_window_size[1] - 1)
            else:
                coords_table[:, :, 0] /= max(Wh - 1, 1)
                coords_table[:, :, 1] /= max(Ww - 1, 1)
            coords_table *= 8
            coords_table = (
                torch.sign(coords_table)
                * torch.log2(torch.abs(coords_table) + 1.0)
                / math.log2(8)
            )
            # Build position index for actual window size
            ch = torch.arange(Wh, device=self.logit_scale.device)
            cw = torch.arange(Ww, device=self.logit_scale.device)
            coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij'))
            coords_flat = coords.view(2, -1)
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel = rel.permute(1, 2, 0).contiguous()
            rel[:, :, 0] += Wh - 1
            rel[:, :, 1] += Ww - 1
            rel[:, :, 0] *= 2 * Ww - 1
            pos_index = rel.sum(-1)

            bias_table = self.cpb_mlp(coords_table).view(-1, self.num_heads)
            relative_position_bias = bias_table[
                pos_index.view(-1)
            ].view(N, N, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C) where N = Wh*Ww
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape

        # Compute QKV with bias
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias, requires_grad=False),
                 self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=math.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale

        # Log-CPB relative position bias (supports dynamic window sizes)
        relative_position_bias = self._compute_position_bias(N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multi-head Self-Attention.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA. Default: 0.
        attn_drop (float): Attention dropout rate. Default: 0.0.
        proj_drop (float): Projection dropout rate. Default: 0.0.
        drop_path (float): Drop path rate. Default: 0.0.
        pad_small_map (bool): Pad small feature maps to window size. Default: False.
        pretrained_window_size (int): Pretrained window size. Default: 0.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pad_small_map: bool = False,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.pad_small_map = pad_small_map

        self.w_msa = WindowMSAV2(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            pretrained_window_size=to_2tuple(pretrained_window_size),
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, f"Input length {L} != H*W ({H}*{W})"

        x = x.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        # Pad or shrink window
        if self.pad_small_map:
            pad_r = (window_size - W % window_size) % window_size
            pad_b = (window_size - H % window_size) % window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            _, Hp, Wp, _ = x.shape
        else:
            Hp, Wp = H, W
            if window_size > Hp:
                window_size = Hp
                shift_size = 0
            if window_size > Wp:
                window_size = Wp
                shift_size = 0

        # Compute attention mask for SW-MSA
        attn_mask = self._compute_attn_mask(Hp, Wp, window_size, shift_size, x.device)

        # Cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Partition windows
        x_windows = self._window_partition(x, window_size)
        # (num_windows*B, window_size*window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.w_msa(x_windows, mask=attn_mask)

        # Merge windows
        x = self._window_reverse(attn_windows, window_size, Hp, Wp)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        if self.pad_small_map and (pad_r > 0 or pad_b > 0):
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop_path(x)
        return x

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        return windows

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        B_nW = windows.shape[0]
        nH = H // window_size
        nW = W // window_size
        B = B_nW // (nH * nW)
        x = windows.view(B, nH, nW, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x

    @staticmethod
    def _compute_attn_mask(H, W, window_size, shift_size, device):
        """Compute attention mask for shifted window attention."""
        if shift_size <= 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition mask
        mask_windows = img_mask.view(
            1, H // window_size, window_size, W // window_size, window_size, 1
        )
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, window_size * window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling (2x).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        norm_layer (type): Normalization layer. Default: nn.LayerNorm.
        is_post_norm (bool): Apply norm after linear. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: type = nn.LayerNorm,
        is_post_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_post_norm = is_post_norm
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        if is_post_norm:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(4 * in_channels)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W

        x = x.view(B, H, W, C)

        # Pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        out_h = (H + pad_h) // 2
        out_w = (W + pad_w) // 2
        x = x.view(B, out_h * out_w, 4 * C)

        if self.is_post_norm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x, (out_h, out_w)
