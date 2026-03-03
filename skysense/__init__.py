"""SkySense: A Multi-Modal Remote Sensing Foundation Model.

Pure PyTorch (2.0+) + HuggingFace Transformers implementation.
No mmcv, mmseg, mmcls, or mmdet dependencies required.
"""

from .configuration_skysense import SkySenseSwinV2Config, SkySenseViTConfig
from .modeling_skysense_swinv2 import (
    SkySenseSwinV2Model,
    SkySenseSwinV2PreTrainedModel,
)
from .modeling_skysense_vit import (
    SkySenseViTModel,
    SkySenseViTPreTrainedModel,
)
from .pipeline import SkySenseFeatureExtractionPipeline

__all__ = [
    "SkySenseSwinV2Config",
    "SkySenseViTConfig",
    "SkySenseSwinV2Model",
    "SkySenseSwinV2PreTrainedModel",
    "SkySenseViTModel",
    "SkySenseViTPreTrainedModel",
    "SkySenseFeatureExtractionPipeline",
]
