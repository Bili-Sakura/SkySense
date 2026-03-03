"""Custom HuggingFace pipeline for SkySense feature extraction."""

from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import Pipeline


class SkySenseFeatureExtractionPipeline(Pipeline):
    """Pipeline for extracting features from remote sensing imagery using SkySense.

    This pipeline takes image tensors (not PIL images) since remote sensing
    data may have non-standard channel counts (e.g., 10 bands for Sentinel-2).

    Example usage::

        from skysense import SkySenseSwinV2Model, SkySenseSwinV2Config
        from skysense.pipeline import SkySenseFeatureExtractionPipeline

        config = SkySenseSwinV2Config(arch='huge', in_channels=3)
        model = SkySenseSwinV2Model(config)
        pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')
        features = pipe(torch.randn(1, 3, 224, 224))
    """

    def _sanitize_parameters(self, **kwargs) -> tuple:
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        if "output_hidden_states" in kwargs:
            forward_kwargs["output_hidden_states"] = kwargs["output_hidden_states"]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Any, **kwargs) -> Dict[str, torch.Tensor]:
        """Preprocess inputs into model-ready tensor.

        Args:
            inputs: Can be a torch.Tensor (B, C, H, W) or numpy array.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.float()
        else:
            raise TypeError(
                f"Expected torch.Tensor or numpy.ndarray, got {type(inputs)}"
            )

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        return {"pixel_values": inputs}

    def _forward(self, model_inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Run model forward pass."""
        pixel_values = model_inputs["pixel_values"]
        output_hidden_states = kwargs.get("output_hidden_states", False)

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        return {"outputs": outputs}

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Post-process model outputs."""
        outputs = model_outputs["outputs"]
        result = {
            "last_hidden_state": outputs.last_hidden_state,
        }
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
        return result
