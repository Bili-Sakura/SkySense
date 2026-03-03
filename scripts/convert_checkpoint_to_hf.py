#!/usr/bin/env python3
"""Convert original SkySense checkpoints to HuggingFace Transformers format.

This script converts pre-trained SkySense weights from the original format
(using mmcv/mmcls/mmseg conventions) to HuggingFace-compatible format that
can be loaded with SkySenseSwinV2Model or SkySenseViTModel.

Usage:
    # Convert HR/RGB backbone (Swin Transformer V2)
    python scripts/convert_checkpoint_to_hf.py \
        --input-path skysense_pretrained.pth \
        --data-type rgb \
        --output-dir ./skysense-swinv2-huge-rgb

    # Convert Sentinel-2 backbone (ViT)
    python scripts/convert_checkpoint_to_hf.py \
        --input-path skysense_pretrained.pth \
        --data-type s2 \
        --output-dir ./skysense-vit-large-s2

    # Convert Sentinel-1 backbone (ViT)
    python scripts/convert_checkpoint_to_hf.py \
        --input-path skysense_pretrained.pth \
        --data-type s1 \
        --output-dir ./skysense-vit-large-s1

    # Convert RGBNIR backbone (Swin Transformer V2 with 4 input channels)
    python scripts/convert_checkpoint_to_hf.py \
        --input-path skysense_pretrained.pth \
        --data-type rgbnir \
        --output-dir ./skysense-swinv2-huge-rgbnir
"""

import argparse
import os
import re
import sys
from collections import OrderedDict

import torch

# Add parent directory to path to import skysense
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skysense import (
    SkySenseSwinV2Config,
    SkySenseSwinV2Model,
    SkySenseViTConfig,
    SkySenseViTModel,
)


# =============================================================================
# Key mapping from original mmcv/mmcls-based checkpoint to HF format
# =============================================================================

def _extract_hr_weights(ckpt):
    """Extract high-resolution (Swin V2) backbone weights from SkySense checkpoint."""
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('backbone_gep.'):
            new_k = k.replace('backbone_gep.', '')
        elif any(k.startswith(p) for p in ('backbone_s1.', 'backbone_s2.', 'fusion.', 'head_gep', 'head')):
            continue
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt


def _extract_s2_weights(ckpt):
    """Extract Sentinel-2 (ViT) backbone weights from SkySense checkpoint."""
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('backbone_s2.'):
            new_k = k.replace('backbone_s2.', '')
        elif any(k.startswith(p) for p in ('backbone_gep.', 'backbone_s1.', 'fusion.', 'head_gep', 'head')):
            continue
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt


def _extract_s1_weights(ckpt):
    """Extract Sentinel-1 (ViT) backbone weights from SkySense checkpoint."""
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('backbone_s1.'):
            new_k = k.replace('backbone_s1.', '')
        elif any(k.startswith(p) for p in ('backbone_gep.', 'backbone_s2.', 'fusion.', 'head_gep', 'head')):
            continue
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt


def _convert_swinv2_keys(state_dict):
    """Convert mmcls Swin V2 keys to HuggingFace SkySenseSwinV2Model keys.

    Key mapping:
        Original (mmcls)                          -> HF (SkySenseSwinV2Model)
        patch_embed.projection.weight             -> patch_embed.projection.weight
        patch_embed.norm.weight                   -> patch_embed.norm.weight
        stages.X.downsample.*                     -> stages.X.downsample.*
        stages.X.blocks.Y.attn.w_msa.*           -> stages.X.blocks.Y.attn.w_msa.*
        stages.X.blocks.Y.norm1.*                 -> stages.X.blocks.Y.norm1.*
        stages.X.blocks.Y.ffn.layers.Z.*         -> stages.X.blocks.Y.ffn.layers.Z.*
        stages.X.blocks.Y.norm2.*                 -> stages.X.blocks.Y.norm2.*
        normN.*                                   -> normN.*

    The key structure is mostly preserved since we reimplemented the same
    architecture. The main differences are:
    - We skip relative_position_index and relative_coords_table (re-built from scratch)
    - We skip mask_token if present
    """
    converted = OrderedDict()
    for k, v in state_dict.items():
        # Skip buffers that are re-initialized
        if 'relative_position_index' in k or 'relative_coords_table' in k:
            continue
        if k == 'mask_token':
            continue
        converted[k] = v
    return converted


def _convert_vit_keys(state_dict):
    """Convert mmcv/mmseg ViT keys to HuggingFace SkySenseViTModel keys.

    Key mapping:
        Original (mmseg)                          -> HF (SkySenseViTModel)
        patch_embed.projection.weight             -> patch_embed.projection.weight
        cls_token                                 -> cls_token
        pos_embed                                 -> pos_embed
        layers.X.norm1.weight                     -> layers.X.norm1.weight
        layers.X.attn.attn.in_proj_weight         -> layers.X.attn.in_proj_weight
        layers.X.attn.attn.in_proj_bias           -> layers.X.attn.in_proj_bias
        layers.X.attn.attn.out_proj.weight        -> layers.X.attn.out_proj.weight
        layers.X.attn.attn.out_proj.bias          -> layers.X.attn.out_proj.bias
        layers.X.ffn.layers.0.0.weight            -> layers.X.ffn.layers.0.weight
        layers.X.ffn.layers.0.0.bias              -> layers.X.ffn.layers.0.bias
        layers.X.ffn.layers.1.weight              -> layers.X.ffn.layers.3.weight
        layers.X.ffn.layers.1.bias                -> layers.X.ffn.layers.3.bias
        ln1.weight                                -> norm.weight
    """
    converted = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        if k == 'ctpe':
            continue

        # Map mmcv MultiheadAttention -> nn.MultiheadAttention
        # mmcv wraps attn as self.attn.attn (MultiheadAttention.attn = nn.MultiheadAttention)
        new_k = re.sub(r'layers\.(\d+)\.attn\.attn\.', r'layers.\1.attn.', new_k)

        # Map mmcv FFN layer indices to nn.Sequential indices
        # mmcv FFN: layers.0.0 = Linear, layers.0.1 = Act, layers.0.2 = Dropout, layers.1 = Linear, layers.2 = Dropout
        # Our FFN:  layers.0 = Linear, layers.1 = Act, layers.2 = Dropout, layers.3 = Linear, layers.4 = Dropout
        new_k = re.sub(r'\.ffn\.layers\.0\.0\.', r'.ffn.layers.0.', new_k)
        new_k = re.sub(r'\.ffn\.layers\.1\.', r'.ffn.layers.3.', new_k)

        # Map final norm
        new_k = new_k.replace('ln1.', 'norm.')

        converted[new_k] = v
    return converted


# =============================================================================
# Main conversion functions
# =============================================================================

def convert_swinv2_checkpoint(ckpt, data_type='rgb'):
    """Convert a SkySense checkpoint to HuggingFace SwinV2 format.

    Args:
        ckpt: Original checkpoint state dict.
        data_type: 'rgb' or 'rgbnir'.

    Returns:
        Tuple of (config, converted_state_dict).
    """
    weights = _extract_hr_weights(ckpt)

    if data_type == 'rgbnir':
        # Copy red band weights to near-infrared band
        proj_key = 'patch_embed.projection.weight'
        if proj_key in weights:
            weights[proj_key] = torch.cat(
                (weights[proj_key],
                 weights[proj_key][:, 0, :, :].unsqueeze(1)),
                dim=1,
            )

    in_channels = 4 if data_type == 'rgbnir' else 3
    config = SkySenseSwinV2Config(arch='huge', in_channels=in_channels)
    converted = _convert_swinv2_keys(weights)
    return config, converted


def convert_vit_checkpoint(ckpt, data_type='s2'):
    """Convert a SkySense checkpoint to HuggingFace ViT format.

    Args:
        ckpt: Original checkpoint state dict.
        data_type: 's2' or 's1'.

    Returns:
        Tuple of (config, converted_state_dict).
    """
    if data_type == 's2':
        weights = _extract_s2_weights(ckpt)
        in_channels = 10
    elif data_type == 's1':
        weights = _extract_s1_weights(ckpt)
        in_channels = 2
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    config = SkySenseViTConfig(in_channels=in_channels)
    converted = _convert_vit_keys(weights)
    return config, converted


def convert_and_save(input_path, data_type, output_dir):
    """Load original checkpoint, convert, and save in HuggingFace format.

    Args:
        input_path: Path to the original SkySense checkpoint.
        data_type: One of 'rgb', 'rgbnir', 's2', 's1'.
        output_dir: Directory to save the HuggingFace model.
    """
    print(f"Loading checkpoint from {input_path}...")
    # Note: weights_only=False is required for legacy checkpoints containing
    # non-tensor objects. Only use with trusted checkpoint sources.
    raw = torch.load(input_path, map_location='cpu', weights_only=False)
    ckpt = raw.get('model', raw)

    if data_type in ('rgb', 'rgbnir'):
        print(f"Converting Swin Transformer V2 backbone ({data_type})...")
        config, state_dict = convert_swinv2_checkpoint(ckpt, data_type)
        model = SkySenseSwinV2Model(config)
    elif data_type in ('s2', 's1'):
        print(f"Converting Vision Transformer backbone ({data_type})...")
        config, state_dict = convert_vit_checkpoint(ckpt, data_type)
        model = SkySenseViTModel(config)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Load weights (strict=False to allow missing buffers like relative_position_index)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load state dict result:\n  missing: {msg.missing_keys}\n  unexpected: {msg.unexpected_keys}")

    # Save in HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SkySense checkpoints to HuggingFace Transformers format"
    )
    parser.add_argument(
        '--input-path', type=str, required=True,
        help='Path to original SkySense checkpoint (.pth)',
    )
    parser.add_argument(
        '--data-type', type=str, required=True,
        choices=['rgb', 's2', 's1', 'rgbnir'],
        help='Data modality type',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for HuggingFace model',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_and_save(args.input_path, args.data_type, args.output_dir)
