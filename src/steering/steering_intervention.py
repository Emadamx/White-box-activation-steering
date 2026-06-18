"""
SteeringIntervention: Applies the learned direction v̂ to suppress deceptive
activations at inference time via orthogonal projection.

The intervention modifies residual stream activations at layer l* by projecting
out the deceptive component:

    h'_t = h_t - α · (h_t · v̂) · v̂

where α ∈ [0, 1] controls the intervention strength.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn

from src.steering.activation_hook import ActivationHook


class SteeringIntervention:
    """
    Inference-time activation steering via orthogonal projection.

    This class registers a forward hook that modifies activations at a
    specified layer to suppress the deceptive direction.

    Args:
        model: The transformer policy agent to steer.
        direction: Unit steering direction v̂ as a 1-D tensor of shape (d_model,).
        layer_idx: Layer index l* at which to apply the intervention.
        alpha: Steering coefficient in [0, 1]. 0 = no intervention, 1 = full projection.
    """

    def __init__(
        self,
        model: nn.Module,
        direction: torch.Tensor,
        layer_idx: int,
        alpha: float = 1.0,
    ) -> None:
        self.model = model
        self.direction = direction / direction.norm()   # ensure unit vector
        self.layer_idx = layer_idx
        self.alpha = alpha
        self._handle = None

    def _project_out(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project out the deceptive direction from activations.

        h shape: (batch, seq_len, d_model) or (batch, d_model)
        """
        v = self.direction.to(h.device)
        # Compute projection coefficient: (h · v̂)
        if h.dim() == 3:
            # (batch, seq_len, d_model)
            proj = torch.einsum("bsd,d->bs", h, v).unsqueeze(-1) * v
        else:
            # (batch, d_model)
            proj = torch.einsum("bd,d->b", h, v).unsqueeze(-1) * v
        return h - self.alpha * proj

    def _make_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Modify first element (residual stream) only
                modified = self._project_out(output[0])
                return (modified,) + output[1:]
            else:
                return self._project_out(output)
        return hook_fn

    @contextmanager
    def active(self):
        """
        Context manager that enables the steering intervention for the duration
        of the enclosed code block. Automatically removes the hook on exit.

        Usage:
            with intervention.active():
                output = agent.act(observation)
        """
        try:
            layers = self._get_layers()
            if self.layer_idx >= len(layers):
                raise ValueError(
                    f"layer_idx={self.layer_idx} exceeds model depth ({len(layers)} layers)"
                )
            self._handle = layers[self.layer_idx].register_forward_hook(self._make_hook())
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

    def _get_layers(self) -> List[nn.Module]:
        for attr in ["layers", "transformer.h", "model.layers", "blocks"]:
            try:
                obj = self.model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise AttributeError("Cannot find transformer layers in model.")
