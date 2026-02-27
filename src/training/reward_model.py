"""
reward_model.py — Reward Model (Bradley-Terry)
===============================================
Theory
------
A reward model R_φ(x, y) ∈ ℝ assigns a scalar score to response y given
prompt x. Trained on human preference pairs (y_w ≻ y_l) using the
Bradley-Terry pairwise ranking loss:

    L_RM = -log σ(R_φ(x, y_w) - R_φ(x, y_l))

This is the reward model used in Stage 2 of RLHF before PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RewardModel(nn.Module):
    """Reward model: transformer backbone + linear scalar head.

    Parameters
    ----------
    backbone : nn.Module
        A HuggingFace model that outputs hidden states (e.g., GPT-2).
    hidden_size : int
        Dimensionality of the backbone's last hidden state.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int = 768) -> None:
        super().__init__()
        self.backbone = backbone
        # Single scalar output — the reward
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar reward for each sequence in the batch.

        Uses the last non-padding token's hidden state as the sequence repr.
        (Same as the "end-of-sequence" pooling strategy.)

        Returns
        -------
        torch.Tensor of shape (B,)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H) — requires output_hidden_states

        # Pool: take the last non-padding token for each example
        if attention_mask is not None:
            # Sum positions to find last real token
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            last_token_idx = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
            )
        # Gather: (B, H)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_idx, last_token_idx]  # (B, H)

        return self.reward_head(pooled).squeeze(-1)  # (B,)


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss.

    L = -log σ(r_chosen - r_rejected)

    Parameters
    ----------
    reward_chosen : (B,)
    reward_rejected : (B,)
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


class RewardModelTrainer:
    """Trains a RewardModel on preference pairs.

    Parameters
    ----------
    model : RewardModel
    tokenizer
    learning_rate : float
    num_epochs : int
    batch_size : int
    """

    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        learning_rate: float = 1e-5,
        num_epochs: int = 1,
        batch_size: int = 4,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.loss_history: List[float] = []

    def train(self, examples: List[Dict[str, str]]) -> List[float]:
        """Train on preference pairs.

        Parameters
        ----------
        examples : list[dict] with keys 'prompt', 'chosen', 'rejected'
        """
        device = next(self.model.parameters()).device
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        losses: List[float] = []

        for epoch in range(self.epochs):
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i: i + self.batch_size]
                chosen_texts = [b["prompt"] + b["chosen"] for b in batch]
                rejected_texts = [b["prompt"] + b["rejected"] for b in batch]

                chosen_enc = self.tokenizer(chosen_texts, return_tensors="pt",
                                            padding=True, truncation=True, max_length=256)
                rejected_enc = self.tokenizer(rejected_texts, return_tensors="pt",
                                              padding=True, truncation=True, max_length=256)

                r_chosen = self.model(
                    chosen_enc["input_ids"].to(device),
                    chosen_enc["attention_mask"].to(device),
                )
                r_rejected = self.model(
                    rejected_enc["input_ids"].to(device),
                    rejected_enc["attention_mask"].to(device),
                )

                loss = bradley_terry_loss(r_chosen, r_rejected)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        self.loss_history = losses
        return losses

    def predict(self, text: str) -> float:
        """Return the scalar reward for a single text."""
        self.model.eval()
        device = next(self.model.parameters()).device
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            reward = self.model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
        return reward.item()


if __name__ == "__main__":
    print("RewardModel — see Notebook 06 for a full runnable example.")
    print("Core: bradley_terry_loss(-log σ(r_w - r_l))")
