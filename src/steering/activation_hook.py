"""
ActivationHook: PyTorch forward hooks for extracting intermediate activations
from transformer policy agents.

Used in the direction-finding pipeline to collect H^c and H^d activation
matrices from cooperative and deceptive rollouts.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ActivationHook:
    """
    Attaches PyTorch forward hooks to specified layers of a transformer model
    and collects their activations during a forward pass.

    Usage:
        hook = ActivationHook(model, layer_indices=[6, 7, 8])
        with hook.capture():
            output = model(input_tensor)
        activations = hook.get_activations()   # {layer_idx: Tensor}

    Args:
        model: The nn.Module to hook into.
        layer_indices: List of layer indices to hook. If None, hooks all layers
                       in model.layers (assumes GPT-2 style architecture).
        hook_point: Which sub-module within each layer to hook.
                    Supported: 'mlp', 'attn', 'residual' (default).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        hook_point: str = "residual",
    ) -> None:
        self.model = model
        self.hook_point = hook_point
        self._activations: Dict[int, torch.Tensor] = {}
        self._handles = []

        # Resolve which layers to hook
        layers = self._get_layers()
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        self.layer_indices = layer_indices
        self._layers = {i: layers[i] for i in layer_indices if i < len(layers)}

    def _get_layers(self) -> List[nn.Module]:
        """Attempt to extract the transformer's layer list."""
        # Try common attribute names across architectures
        for attr in ["layers", "transformer.h", "model.layers", "blocks"]:
            try:
                obj = self.model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise AttributeError(
            "Cannot automatically find transformer layers. "
            "Pass a model with a .layers attribute, or subclass ActivationHook."
        )

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output may be a tuple (e.g. attention layers return (attn_out, attn_weights))
            activation = output[0] if isinstance(output, tuple) else output
            # Store the last token's residual: shape (batch, seq_len, d_model) → (batch, d_model)
            self._activations[layer_idx] = activation[:, -1, :].detach().cpu()
        return hook_fn

    @contextmanager
    def capture(self):
        """Context manager: registers hooks on enter, removes them on exit."""
        self._activations = {}
        self._handles = []
        try:
            for idx, layer in self._layers.items():
                handle = layer.register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return collected activations as {layer_idx: Tensor(batch, d_model)}."""
        return dict(self._activations)

    def get_activation(self, layer_idx: int) -> torch.Tensor:
        """Return activations for a specific layer."""
        if layer_idx not in self._activations:
            raise KeyError(f"No activations captured for layer {layer_idx}. "
                           f"Available: {list(self._activations.keys())}")
        return self._activations[layer_idx]
