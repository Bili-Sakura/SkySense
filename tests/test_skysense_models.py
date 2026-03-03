"""Tests for SkySense HuggingFace models (no checkpoint required)."""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skysense import (
    SkySenseFeatureExtractionPipeline,
    SkySenseSwinV2Config,
    SkySenseSwinV2Model,
    SkySenseViTConfig,
    SkySenseViTModel,
)
from skysense.modeling_utils import (
    DropPath,
    FFN,
    PatchEmbed,
    PatchMerging,
    ShiftWindowMSA,
    WindowMSAV2,
    to_2tuple,
)


# ============================================================================
# Utility tests
# ============================================================================

class TestUtils:
    def test_to_2tuple(self):
        assert to_2tuple(4) == (4, 4)
        assert to_2tuple((3, 5)) == (3, 5)
        assert to_2tuple([2, 7]) == (2, 7)

    def test_drop_path_no_drop(self):
        dp = DropPath(0.0)
        x = torch.randn(2, 4)
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_eval(self):
        dp = DropPath(0.5)
        dp.eval()
        x = torch.randn(2, 4)
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_train(self):
        dp = DropPath(0.5)
        dp.train()
        x = torch.randn(100, 4)
        out = dp(x)
        # Some paths should be dropped (zeros)
        assert out.shape == x.shape


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed(in_channels=3, embed_dims=96, kernel_size=4, stride=4)
        x = torch.randn(2, 3, 32, 32)
        out, hw = pe(x)
        assert hw == (8, 8)
        assert out.shape == (2, 64, 96)

    def test_with_norm(self):
        pe = PatchEmbed(
            in_channels=3, embed_dims=96, kernel_size=4, stride=4,
            norm_layer=torch.nn.LayerNorm,
        )
        x = torch.randn(1, 3, 16, 16)
        out, hw = pe(x)
        assert out.shape == (1, 16, 96)

    def test_init_out_size(self):
        pe = PatchEmbed(
            in_channels=3, embed_dims=96, kernel_size=4, stride=4,
            input_size=224,
        )
        assert pe.init_out_size == (56, 56)


class TestFFN:
    def test_output_shape(self):
        ffn = FFN(embed_dims=64, feedforward_channels=256, add_identity=True)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_no_identity(self):
        ffn = FFN(embed_dims=32, feedforward_channels=128, add_identity=False)
        x = torch.randn(1, 5, 32)
        out = ffn(x)
        assert out.shape == x.shape


class TestWindowMSAV2:
    def test_output_shape(self):
        msa = WindowMSAV2(
            embed_dims=64, num_heads=4, window_size=(4, 4),
        )
        x = torch.randn(8, 16, 64)  # 8 windows, 4x4=16 tokens each
        out = msa(x)
        assert out.shape == (8, 16, 64)

    def test_with_mask(self):
        msa = WindowMSAV2(
            embed_dims=64, num_heads=4, window_size=(4, 4),
        )
        x = torch.randn(8, 16, 64)
        mask = torch.zeros(8, 16, 16)
        out = msa(x, mask=mask)
        assert out.shape == (8, 16, 64)


class TestPatchMerging:
    def test_output_shape(self):
        pm = PatchMerging(in_channels=96, out_channels=192)
        x = torch.randn(2, 64, 96)  # 8x8 spatial, 96 channels
        out, hw = pm(x, (8, 8))
        assert hw == (4, 4)
        assert out.shape == (2, 16, 192)


class TestShiftWindowMSA:
    def test_no_shift(self):
        swmsa = ShiftWindowMSA(
            embed_dims=64, num_heads=4, window_size=4, shift_size=0,
        )
        x = torch.randn(1, 64, 64)  # 8x8 spatial
        out = swmsa(x, (8, 8))
        assert out.shape == (1, 64, 64)

    def test_with_shift(self):
        swmsa = ShiftWindowMSA(
            embed_dims=64, num_heads=4, window_size=4, shift_size=2,
        )
        x = torch.randn(1, 64, 64)
        out = swmsa(x, (8, 8))
        assert out.shape == (1, 64, 64)


# ============================================================================
# Config tests
# ============================================================================

class TestSwinV2Config:
    def test_default_config(self):
        config = SkySenseSwinV2Config()
        assert config.arch == "huge"
        assert config.embed_dims == 352
        assert config.depths == [2, 2, 18, 2]
        assert config.num_heads == [8, 16, 32, 64]

    def test_tiny_config(self):
        config = SkySenseSwinV2Config(arch='tiny')
        assert config.embed_dims == 96
        assert config.depths == [2, 2, 6, 2]

    def test_serialization(self):
        config = SkySenseSwinV2Config(arch='tiny', in_channels=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = SkySenseSwinV2Config.from_pretrained(tmpdir)
            assert loaded.arch == 'tiny'
            assert loaded.in_channels == 4
            assert loaded.embed_dims == 96


class TestViTConfig:
    def test_default_config(self):
        config = SkySenseViTConfig()
        assert config.embed_dims == 1024
        assert config.num_layers == 24
        assert config.in_channels == 10

    def test_s1_config(self):
        config = SkySenseViTConfig(in_channels=2)
        assert config.in_channels == 2

    def test_serialization(self):
        config = SkySenseViTConfig(in_channels=2, num_layers=12)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = SkySenseViTConfig.from_pretrained(tmpdir)
            assert loaded.in_channels == 2
            assert loaded.num_layers == 12


# ============================================================================
# Model tests (using tiny configs for speed)
# ============================================================================

class TestSwinV2Model:
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny SwinV2 model for testing."""
        config = SkySenseSwinV2Config(
            arch='tiny',
            img_size=32,
            window_size=4,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.0,
        )
        model = SkySenseSwinV2Model(config)
        model.eval()
        return model

    def test_forward_shape(self, tiny_model):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            outputs = tiny_model(x, return_dict=False)
        assert len(outputs) == 4
        # Stage 0: 8x8, 96 channels
        assert outputs[0].shape == (1, 96, 8, 8)
        # Stage 1: 4x4, 192 channels
        assert outputs[1].shape == (1, 192, 4, 4)
        # Stage 2: 2x2, 384 channels
        assert outputs[2].shape == (1, 384, 2, 2)
        # Stage 3: 1x1, 768 channels
        assert outputs[3].shape == (1, 768, 1, 1)

    def test_forward_return_dict(self, tiny_model):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            outputs = tiny_model(x, return_dict=True)
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state is not None

    def test_forward_hidden_states(self, tiny_model):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            outputs = tiny_model(x, output_hidden_states=True, return_dict=True)
        assert outputs.hidden_states is not None
        assert len(outputs.hidden_states) == 4

    def test_single_out_index(self):
        config = SkySenseSwinV2Config(
            arch='tiny', img_size=32, window_size=4, out_indices=(3,),
        )
        model = SkySenseSwinV2Model(config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            outputs = model(x, return_dict=False)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 768, 1, 1)

    def test_pad_small_map(self):
        config = SkySenseSwinV2Config(
            arch='tiny', img_size=30, window_size=4, out_indices=(3,),
            pad_small_map=True,
        )
        model = SkySenseSwinV2Model(config)
        model.eval()
        x = torch.randn(1, 3, 30, 30)
        with torch.no_grad():
            outputs = model(x, return_dict=False)
        assert len(outputs) == 1

    def test_save_load(self):
        config = SkySenseSwinV2Config(arch='tiny', img_size=32, window_size=4)
        model = SkySenseSwinV2Model(config)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x, return_dict=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SkySenseSwinV2Model.from_pretrained(tmpdir)
            loaded.eval()
            with torch.no_grad():
                out2 = loaded(x, return_dict=False)

        for o1, o2 in zip(out1, out2):
            assert torch.allclose(o1, o2, atol=1e-5)


class TestViTModel:
    @pytest.fixture
    def small_model(self):
        """Create a small ViT model for testing."""
        config = SkySenseViTConfig(
            img_size=16,
            patch_size=4,
            in_channels=3,
            embed_dims=64,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            out_indices=(1,),
            drop_path_rate=0.0,
        )
        model = SkySenseViTModel(config)
        model.eval()
        return model

    def test_forward_shape(self, small_model):
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            outputs = small_model(x, return_dict=False)
        assert len(outputs) == 1
        # (B, C, H, W) = (1, 64, 4, 4)
        assert outputs[0].shape == (1, 64, 4, 4)

    def test_forward_return_dict(self, small_model):
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            outputs = small_model(x, return_dict=True)
        assert hasattr(outputs, 'last_hidden_state')

    def test_s2_config(self):
        """Test with Sentinel-2 default config (10 channels)."""
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=10,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()
        x = torch.randn(1, 10, 16, 16)
        with torch.no_grad():
            outputs = model(x, return_dict=False)
        assert len(outputs) == 1

    def test_s1_config(self):
        """Test with Sentinel-1 config (2 channels)."""
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=2,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()
        x = torch.randn(1, 2, 16, 16)
        with torch.no_grad():
            outputs = model(x, return_dict=False)
        assert len(outputs) == 1

    def test_with_cls_token(self):
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=3,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
            with_cls_token=True, output_cls_token=True,
        )
        model = SkySenseViTModel(config)
        model.eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            outputs = model(x, return_dict=False)
        # output_cls_token returns [feature_map, cls_token]
        assert isinstance(outputs[0], list)
        assert len(outputs[0]) == 2

    def test_save_load(self):
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=3,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out1 = model(x, return_dict=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SkySenseViTModel.from_pretrained(tmpdir)
            loaded.eval()
            with torch.no_grad():
                out2 = loaded(x, return_dict=False)

        for o1, o2 in zip(out1, out2):
            assert torch.allclose(o1, o2, atol=1e-5)


# ============================================================================
# Pipeline tests
# ============================================================================

class TestPipeline:
    def test_swinv2_pipeline(self):
        config = SkySenseSwinV2Config(
            arch='tiny', img_size=32, window_size=4,
        )
        model = SkySenseSwinV2Model(config)
        model.eval()

        pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')
        x = torch.randn(1, 3, 32, 32)
        result = pipe(x)
        assert 'last_hidden_state' in result
        assert result['last_hidden_state'] is not None

    def test_vit_pipeline(self):
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=10,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()

        pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')
        x = torch.randn(1, 10, 16, 16)
        result = pipe(x)
        assert 'last_hidden_state' in result

    def test_numpy_input(self):
        import numpy as np

        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=3,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()

        pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')
        x = np.random.randn(1, 3, 16, 16).astype(np.float32)
        result = pipe(x)
        assert 'last_hidden_state' in result

    def test_3d_input_auto_expand(self):
        config = SkySenseViTConfig(
            img_size=16, patch_size=4, in_channels=3,
            embed_dims=64, num_layers=2, num_heads=4, mlp_ratio=2,
        )
        model = SkySenseViTModel(config)
        model.eval()

        pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')
        x = torch.randn(3, 16, 16)  # 3D input
        result = pipe(x)
        assert 'last_hidden_state' in result


# ============================================================================
# Conversion tests (without actual checkpoint)
# ============================================================================

class TestConversion:
    def test_swinv2_key_conversion(self):
        from scripts.convert_checkpoint_to_hf import _convert_swinv2_keys

        state_dict = {
            'patch_embed.projection.weight': torch.randn(96, 3, 4, 4),
            'stages.0.blocks.0.norm1.weight': torch.randn(96),
            'stages.0.blocks.0.attn.w_msa.relative_position_index': torch.zeros(16, 16),
            'stages.0.blocks.0.attn.w_msa.relative_coords_table': torch.zeros(1, 49, 2),
            'mask_token': torch.randn(1, 1, 96),
        }
        converted = _convert_swinv2_keys(state_dict)
        assert 'patch_embed.projection.weight' in converted
        assert 'stages.0.blocks.0.norm1.weight' in converted
        # Buffers should be skipped
        assert 'stages.0.blocks.0.attn.w_msa.relative_position_index' not in converted
        assert 'stages.0.blocks.0.attn.w_msa.relative_coords_table' not in converted
        assert 'mask_token' not in converted

    def test_vit_key_conversion(self):
        from scripts.convert_checkpoint_to_hf import _convert_vit_keys

        state_dict = {
            'cls_token': torch.randn(1, 1, 64),
            'pos_embed': torch.randn(1, 17, 64),
            'layers.0.attn.attn.in_proj_weight': torch.randn(192, 64),
            'layers.0.attn.attn.in_proj_bias': torch.randn(192),
            'layers.0.attn.attn.out_proj.weight': torch.randn(64, 64),
            'layers.0.attn.attn.out_proj.bias': torch.randn(64),
            'layers.0.ffn.layers.0.0.weight': torch.randn(256, 64),
            'layers.0.ffn.layers.0.0.bias': torch.randn(256),
            'layers.0.ffn.layers.1.weight': torch.randn(64, 256),
            'layers.0.ffn.layers.1.bias': torch.randn(64),
            'ln1.weight': torch.randn(64),
            'ln1.bias': torch.randn(64),
            'ctpe': torch.randn(10),
        }
        converted = _convert_vit_keys(state_dict)
        # Check attention key mapping
        assert 'layers.0.attn.in_proj_weight' in converted
        assert 'layers.0.attn.out_proj.weight' in converted
        # Check FFN key mapping
        assert 'layers.0.ffn.layers.0.weight' in converted
        assert 'layers.0.ffn.layers.3.weight' in converted
        # Check final norm mapping
        assert 'norm.weight' in converted
        assert 'norm.bias' in converted
        # ctpe should be skipped
        assert 'ctpe' not in converted

    def test_extract_hr_weights(self):
        from scripts.convert_checkpoint_to_hf import _extract_hr_weights

        ckpt = {
            'backbone_gep.patch_embed.weight': torch.randn(96, 3, 4, 4),
            'backbone_s1.some_param': torch.randn(10),
            'backbone_s2.some_param': torch.randn(10),
            'fusion.layer': torch.randn(10),
            'head_gep.fc': torch.randn(10),
            'head.fc': torch.randn(10),
        }
        result = _extract_hr_weights(ckpt)
        assert 'patch_embed.weight' in result
        assert len(result) == 1

    def test_extract_s2_weights(self):
        from scripts.convert_checkpoint_to_hf import _extract_s2_weights

        ckpt = {
            'backbone_gep.patch_embed.weight': torch.randn(96, 3, 4, 4),
            'backbone_s2.cls_token': torch.randn(1, 1, 1024),
            'backbone_s1.cls_token': torch.randn(1, 1, 1024),
            'fusion.layer': torch.randn(10),
        }
        result = _extract_s2_weights(ckpt)
        assert 'cls_token' in result
        assert len(result) == 1

    def test_extract_s1_weights(self):
        from scripts.convert_checkpoint_to_hf import _extract_s1_weights

        ckpt = {
            'backbone_gep.patch_embed.weight': torch.randn(96, 3, 4, 4),
            'backbone_s2.cls_token': torch.randn(1, 1, 1024),
            'backbone_s1.pos_embed': torch.randn(1, 257, 1024),
            'fusion.layer': torch.randn(10),
        }
        result = _extract_s1_weights(ckpt)
        assert 'pos_embed' in result
        assert len(result) == 1

    def test_full_conversion_swinv2_synthetic(self):
        """Test full conversion with synthetic checkpoint."""
        from scripts.convert_checkpoint_to_hf import convert_swinv2_checkpoint

        config = SkySenseSwinV2Config(arch='tiny', img_size=32, window_size=4)
        model = SkySenseSwinV2Model(config)

        # Create a synthetic "original" checkpoint
        original_ckpt = {}
        for name, param in model.named_parameters():
            original_ckpt['backbone_gep.' + name] = param.data.clone()
        # Add extra keys that should be filtered
        original_ckpt['backbone_s1.fake'] = torch.randn(10)
        original_ckpt['fusion.fake'] = torch.randn(10)

        config_out, converted = convert_swinv2_checkpoint(original_ckpt, 'rgb')
        assert config_out.arch == 'huge'  # default huge config

    def test_full_conversion_vit_synthetic(self):
        """Test full conversion with synthetic checkpoint."""
        from scripts.convert_checkpoint_to_hf import convert_vit_checkpoint

        # Create a synthetic "original" checkpoint with mmcv-style keys
        original_ckpt = {
            'backbone_s2.cls_token': torch.randn(1, 1, 1024),
            'backbone_s2.pos_embed': torch.randn(1, 257, 1024),
            'backbone_gep.fake': torch.randn(10),
        }

        config_out, converted = convert_vit_checkpoint(original_ckpt, 's2')
        assert config_out.in_channels == 10
        assert 'cls_token' in converted
        assert 'pos_embed' in converted
