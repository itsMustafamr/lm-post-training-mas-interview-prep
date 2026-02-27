"""
dpo_trainer.py — Direct Preference Optimization Trainer
=========================================================
Theory
------
DPO (Rafailov et al., 2023) eliminates the explicit reward model by
reparameterizing RLHF's objective directly in terms of the policy.

DPO Loss:
    L_DPO(π_θ) = -E_{(x,y_w,y_l)~D}[ log σ(
        β · log(π_θ(y_w|x)/π_ref(y_w|x))
      - β · log(π_θ(y_l|x)/π_ref(y_l|x))
    )]

Where:
  - y_w = "chosen" (preferred) response
  - y_l = "rejected" (dispreferred) response
  - π_ref = frozen reference policy (the SFT model)
  - β = KL-penalty coefficient (higher → stay closer to reference)
  - σ = sigmoid function

Key insight: The optimal reward is implicitly defined by the ratio
r*(x,y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class DPOConfig:
    """Hyperparameters for DPO training.

    Attributes
    ----------
    beta : float
        KL penalty. Higher β → policy stays closer to reference.
        Typical range: 0.01 – 0.5. Paper default: 0.1.
    learning_rate : float
    num_epochs : int
    batch_size : int
    max_length : int
    """

    beta: float = 0.1
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 2
    max_length: int = 256
    logging_steps: int = 5
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core DPO functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of log-probabilities for the *response* tokens only.

    Parameters
    ----------
    model : causal LM
    input_ids : (B, L)
    attention_mask : (B, L)
    labels : (B, L) — -100 for prompt tokens (masked), actual ids for response

    Returns
    -------
    torch.Tensor of shape (B,)
        Sum of log p(y_t | y_{<t}, x) over response tokens.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

    # Shift: logits at position t predict token at position t+1
    shift_logits = logits[:, :-1, :]   # (B, L-1, V)
    shift_labels = labels[:, 1:]        # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)

    # Gather log-prob of the actual token
    # Shape: (B, L-1)
    token_log_probs = torch.gather(
        log_probs, dim=2, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Mask out prompt positions (where shift_labels == -100)
    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DPO loss.

    L_DPO = -log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                       - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) )

    Parameters
    ----------
    policy_chosen_logps : (B,)  — log p_θ(y_w | x)
    policy_rejected_logps : (B,)
    ref_chosen_logps : (B,)     — log p_ref(y_w | x)
    ref_rejected_logps : (B,)
    beta : float

    Returns
    -------
    (loss, reward_accuracy, reward_margin) : all shape ()
    """
    # Implicit reward for each response
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss = -log σ(r_w - r_l)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, reward_accuracy, reward_margin


class PreferencePairDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) triples for DPO."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        full = prompt + response + self.tokenizer.eos_token
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full, add_special_tokens=False)[: self.max_length]
        n = len(full_ids)
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full((n,), -100, dtype=torch.long)
        resp_start = min(len(prompt_ids), n)
        labels[resp_start:] = input_ids[resp_start:]
        attention_mask = torch.ones(n, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        chosen = self._encode(ex["prompt"], ex["chosen"])
        rejected = self._encode(ex["prompt"], ex["rejected"])
        return {"chosen": chosen, "rejected": rejected}


class DPOTrainer:
    """Trains a policy model using DPO with a frozen reference model.

    Parameters
    ----------
    policy : nn.Module  — the model being trained
    reference : nn.Module  — frozen SFT model
    tokenizer
    config : DPOConfig
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        tokenizer,
        config: Optional[DPOConfig] = None,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()
        # Freeze reference model
        for p in self.reference.parameters():
            p.requires_grad = False
        self.loss_history: List[float] = []

    def train(self, examples: List[Dict[str, str]]) -> List[float]:
        """Train on preference pairs.

        Parameters
        ----------
        examples : list[dict] with keys 'prompt', 'chosen', 'rejected'
        """
        torch.manual_seed(self.config.seed)
        dataset = PreferencePairDataset(examples, self.tokenizer, self.config.max_length)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        device = next(self.policy.parameters()).device
        self.policy.train()
        self.reference.eval()
        losses: List[float] = []

        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(tqdm(loader, desc=f"DPO Epoch {epoch+1}")):
                def to_device(d: Dict) -> Dict:
                    return {k: v.to(device) for k, v in d.items()}

                chosen = to_device(batch["chosen"])
                rejected = to_device(batch["rejected"])

                # Policy log-probs
                pol_chosen = compute_log_probs(self.policy, **chosen)
                pol_rejected = compute_log_probs(self.policy, **rejected)

                # Reference log-probs (no grad)
                with torch.no_grad():
                    ref_chosen = compute_log_probs(self.reference, **chosen)
                    ref_rejected = compute_log_probs(self.reference, **rejected)

                loss, acc, margin = dpo_loss(
                    pol_chosen, pol_rejected, ref_chosen, ref_rejected,
                    beta=self.config.beta,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())
                if (step + 1) % self.config.logging_steps == 0:
                    print(f"  Step {step+1} | loss={loss.item():.4f} "
                          f"| acc={acc.item():.3f} | margin={margin.item():.3f}")

        self.loss_history = losses
        return losses


if __name__ == "__main__":
    print("DPOTrainer — see Notebook 03 for a full runnable example.")
    print("Core functions (compute_log_probs, dpo_loss) are importable standalone.")
