"""
TransformerAgent: A small transformer-based policy agent for the Deceptive Coin Game.

Architecture: Embedding → N × TransformerBlock → linear head → action logits.
Designed to be small enough to run on a personal GPU while having a meaningful
residual stream for activation steering experiments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(attn_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class TransformerAgent(nn.Module):
    """
    Small transformer policy agent for multi-agent RL experiments.

    Args:
        obs_dim: Observation vector size (flattened grid + monitoring signal).
        n_actions: Number of discrete actions (default: 5 — up/down/left/right/stay).
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.d_model = d_model

        # Project obs to d_model (treat as single-token sequence)
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Transformer body — exposed as .layers for hook compatibility
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Policy head
        self.ln_final = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, n_actions)

        # Value head (for actor-critic RL)
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            (action_logits, value) each of shape (batch, n_actions) and (batch, 1).
        """
        # Treat observation as a single-token sequence: (batch, 1, d_model)
        x = self.input_proj(obs).unsqueeze(1)

        for block in self.layers:
            x = block(x)

        # Use the single token's representation
        x = self.ln_final(x[:, 0, :])   # (batch, d_model)

        logits = self.policy_head(x)     # (batch, n_actions)
        value = self.value_head(x)       # (batch, 1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """
        Sample or argmax an action from the policy.

        Args:
            obs: Single observation of shape (obs_dim,) or (1, obs_dim).
            deterministic: If True, take argmax. If False, sample.

        Returns:
            Action integer.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(obs)
            if deterministic:
                return logits.argmax(dim=-1).item()
            else:
                return torch.distributions.Categorical(logits=logits).sample().item()
