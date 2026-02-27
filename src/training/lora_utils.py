"""
lora_utils.py — LoRA & PEFT Utilities
=======================================
Theory
------
LoRA (Low-Rank Adaptation) reparameterizes a weight update ΔW as:

    ΔW = B · A    where  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r ≪ min(d, k)

During forward pass:  h = W₀x + (α/r) · B · A · x

Key properties:
  - Only A and B are trained; W₀ stays frozen.
  - Parameter reduction: r(d+k) vs d·k  (e.g., r=8, d=k=768 → 12,288 vs 589,824)
  - At inference, ΔW can be merged into W₀ with zero extra cost.
  - Multiple adapters can share the same frozen base model.

Reference: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Single LoRA adapter wrapping a frozen linear layer.

    Forward pass: output = frozen_linear(x) + (alpha/r) * B(A(x))

    Parameters
    ----------
    in_features : int
        Input dimensionality (k).
    out_features : int
        Output dimensionality (d).
    r : int
        LoRA rank. Lower = fewer parameters. Typical values: 4, 8, 16.
    alpha : float
        Scaling factor. Effective lr ∝ alpha/r.
    dropout : float
        Dropout applied to the low-rank path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # The (α/r) scalar

        # A: initialized with random Gaussian (following the paper)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        # B: initialized to zero so ΔW = 0 at training start (no perturbation)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Frozen base weights (set externally)
        self.base_weight: Optional[torch.Tensor] = None
        self.base_bias: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W₀x + scaling · B(A(x))."""
        # Base forward
        if self.base_weight is not None:
            base_out = nn.functional.linear(x, self.base_weight, self.base_bias)
        else:
            base_out = torch.zeros(*x.shape[:-1], self.lora_B.out_features,
                                   device=x.device, dtype=x.dtype)

        # LoRA forward: B(A(dropout(x)))
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out

    def get_delta_weight(self) -> torch.Tensor:
        """Return ΔW = scaling · B · A for merging into the base weight."""
        return self.scaling * (self.lora_B.weight @ self.lora_A.weight)

    def trainable_parameters(self) -> int:
        """Count the trainable parameters (A + B only)."""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


def apply_lora_to_linear(
    linear: nn.Linear,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRALayer:
    """Wrap an existing nn.Linear with a LoRALayer.

    The original linear weights are frozen; only A and B are trainable.

    Parameters
    ----------
    linear : nn.Linear
        The existing linear layer to adapt.
    r, alpha, dropout : LoRA hyperparameters.

    Returns
    -------
    LoRALayer
        A new LoRALayer that replicates `linear` plus a trainable LoRA path.
    """
    lora = LoRALayer(
        in_features=linear.in_features,
        out_features=linear.out_features,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    # Store frozen base weights
    lora.base_weight = linear.weight.data.clone()
    lora.base_bias = linear.bias.data.clone() if linear.bias is not None else None

    # Freeze base weights (they are stored as plain tensors, not parameters)
    return lora


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Returns
    -------
    (total, trainable) : tuple[int, int]
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters in a model (useful before applying LoRA)."""
    for param in model.parameters():
        param.requires_grad = False


if __name__ == "__main__":
    # Demo: compare parameter counts for full fine-tuning vs LoRA
    d, k, r = 768, 768, 8
    full_params = d * k
    lora_params = r * (d + k)
    reduction = (1 - lora_params / full_params) * 100

    print(f"Full fine-tuning parameters:  {full_params:,}")
    print(f"LoRA parameters (r={r}):       {lora_params:,}")
    print(f"Parameter reduction:           {reduction:.1f}%")

    # Functional demo
    linear = nn.Linear(768, 768)
    freeze_base_model(linear)
    lora_layer = apply_lora_to_linear(linear, r=r, alpha=16.0)

    x = torch.randn(4, 32, 768)  # (batch, seq, hidden)
    out = lora_layer(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"ΔW shape:     {lora_layer.get_delta_weight().shape}")
    print(f"Trainable parameters in layer: {lora_layer.trainable_parameters():,}")
