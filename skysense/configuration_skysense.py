"""Configuration classes for SkySense models."""

from transformers import PretrainedConfig


class SkySenseSwinV2Config(PretrainedConfig):
    """Configuration class for SkySense Swin Transformer V2 backbone.

    This model handles high-resolution optical imagery (RGB/RGBNIR).

    Args:
        arch (str): Architecture variant. One of 'tiny', 'small', 'base',
            'large', 'huge', 'giant'. Default: 'huge'.
        img_size (int): Input image size. Default: 224.
        patch_size (int): Patch size. Default: 4.
        in_channels (int): Number of input channels. Default: 3.
        window_size (int or list): Window size for each stage. Default: 8.
        drop_rate (float): Dropout rate after embedding. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        out_indices (list): Output indices from stages. Default: [3].
        use_abs_pos_embed (bool): Use absolute position embedding. Default: False.
        with_cp (bool): Use gradient checkpointing. Default: False.
        pad_small_map (bool): Pad small maps to window size. Default: False.
        pretrained_window_sizes (list): Pretrained window sizes. Default: [0, 0, 0, 0].
        is_post_norm_downsample (bool): Use post-norm in downsample. Default: True.
    """

    model_type = "skysense_swinv2"

    arch_zoo = {
        'tiny':  {'embed_dims': 96,  'depths': [2, 2, 6, 2],  'num_heads': [3, 6, 12, 24],  'extra_norm_every_n_blocks': 0},
        'small': {'embed_dims': 96,  'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24],  'extra_norm_every_n_blocks': 0},
        'base':  {'embed_dims': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32],  'extra_norm_every_n_blocks': 0},
        'large': {'embed_dims': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'extra_norm_every_n_blocks': 0},
        'huge':  {'embed_dims': 352, 'depths': [2, 2, 18, 2], 'num_heads': [8, 16, 32, 64], 'extra_norm_every_n_blocks': 6},
        'giant': {'embed_dims': 512, 'depths': [2, 2, 42, 4], 'num_heads': [16, 32, 64, 128], 'extra_norm_every_n_blocks': 6},
    }

    def __init__(
        self,
        arch="huge",
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=8,
        drop_rate=0.0,
        drop_path_rate=0.1,
        out_indices=(3,),
        use_abs_pos_embed=False,
        with_cp=False,
        pad_small_map=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        is_post_norm_downsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(arch, str):
            arch = arch.lower()
            if arch not in self.arch_zoo:
                raise ValueError(f"Unknown arch '{arch}'. Choose from {list(self.arch_zoo.keys())}")
            arch_settings = self.arch_zoo[arch]
        else:
            arch_settings = arch

        self.arch = arch
        self.embed_dims = arch_settings['embed_dims']
        self.depths = arch_settings['depths']
        self.num_heads = arch_settings['num_heads']
        self.extra_norm_every_n_blocks = arch_settings['extra_norm_every_n_blocks']

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.out_indices = list(out_indices)
        self.use_abs_pos_embed = use_abs_pos_embed
        self.with_cp = with_cp
        self.pad_small_map = pad_small_map
        self.pretrained_window_sizes = list(pretrained_window_sizes)
        self.is_post_norm_downsample = is_post_norm_downsample


class SkySenseViTConfig(PretrainedConfig):
    """Configuration class for SkySense Vision Transformer backbone.

    This model handles Sentinel-2 multispectral and Sentinel-1 SAR imagery.

    Args:
        img_size (int): Input image size. Default: 64.
        patch_size (int): Patch size. Default: 4.
        in_channels (int): Number of input channels.
            10 for Sentinel-2, 2 for Sentinel-1. Default: 10.
        embed_dims (int): Embedding dimension. Default: 1024.
        num_layers (int): Number of transformer layers. Default: 24.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (int): MLP hidden dim ratio. Default: 4.
        out_indices (list): Output indices. Default: [-1].
        qkv_bias (bool): QKV bias. Default: True.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.3.
        with_cls_token (bool): Use CLS token. Default: True.
        output_cls_token (bool): Output CLS token. Default: False.
        patch_norm (bool): Norm in patch embed. Default: False.
        final_norm (bool): Final layer norm. Default: False.
        with_cp (bool): Use gradient checkpointing. Default: False.
    """

    model_type = "skysense_vit"

    def __init__(
        self,
        img_size=64,
        patch_size=4,
        in_channels=10,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(-1,),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        with_cls_token=True,
        output_cls_token=False,
        patch_norm=False,
        final_norm=False,
        with_cp=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_indices = list(out_indices)
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.patch_norm = patch_norm
        self.final_norm = final_norm
        self.with_cp = with_cp
