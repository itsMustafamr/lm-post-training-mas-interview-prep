"""
grpo_trainer.py — Group Relative Policy Optimization (GRPO) Trainer
=====================================================================
Theory
------
GRPO (DeepSeek-Math, 2024) removes PPO's value network by estimating
advantages using a *group* of G sampled completions per prompt.

Algorithm per training step:
  1. For each prompt x, sample G completions {y_1,...,y_G} from π_old.
  2. Score each completion: r_i = Reward(x, y_i)  (e.g., binary correctness)
  3. Compute group-normalized advantage:
       A_i = (r_i - mean(r)) / (std(r) + ε)
  4. Update policy with clipped surrogate objective + KL penalty:
       L_GRPO = -E[ min(π_θ/π_old · A, clip(π_θ/π_old, 1-ε, 1+ε) · A) ]
               + β · KL(π_θ || π_ref)

Key advantages over PPO:
  - No value function to train (50% fewer parameters to update)
  - More stable for math tasks with verifiable binary rewards
  - Used in DeepSeek-R1 to achieve SOTA math reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """GRPO hyperparameters.

    Attributes
    ----------
    group_size : int
        G — completions sampled per prompt. Larger G = more stable advantages.
    epsilon_clip : float
        PPO-style clipping parameter ε.
    kl_coeff : float
        β — KL penalty coefficient.
    learning_rate : float
    num_epochs : int
    max_new_tokens : int
    temperature : float
        Sampling temperature for the group completions.
    """

    group_size: int = 8
    epsilon_clip: float = 0.2
    kl_coeff: float = 0.04
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.8
    advantage_epsilon: float = 1e-8
    logging_steps: int = 10
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core GRPO functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards within each group to produce advantages.

    A_i = (r_i - mean(r)) / (std(r) + ε)

    Parameters
    ----------
    rewards : (G,) or (B, G)
        Scalar rewards for each completion in the group.
    epsilon : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor of same shape — normalized advantages.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + epsilon)


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    epsilon_clip: float = 0.2,
    kl_coeff: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the GRPO clipped surrogate loss.

    L = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + β·KL

    Parameters
    ----------
    policy_log_probs : (B,) — log π_θ(y|x)
    old_log_probs : (B,)    — log π_old(y|x)  (detached)
    advantages : (B,)
    ref_log_probs : (B,), optional — log π_ref(y|x) for KL term
    epsilon_clip : float
    kl_coeff : float

    Returns
    -------
    (loss, metrics_dict)
    """
    # Policy ratio: π_θ / π_old
    log_ratio = policy_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantages
    surrogate_loss = -torch.min(surr1, surr2).mean()

    # KL penalty (forward KL: π_θ log(π_θ/π_ref))
    kl_loss = torch.tensor(0.0, device=policy_log_probs.device)
    if ref_log_probs is not None:
        kl_loss = (policy_log_probs - ref_log_probs.detach()).mean()

    total_loss = surrogate_loss + kl_coeff * kl_loss

    metrics = {
        "surrogate_loss": surrogate_loss.item(),
        "kl_loss": kl_loss.item(),
        "mean_ratio": ratio.mean().item(),
        "mean_advantage": advantages.mean().item(),
    }
    return total_loss, metrics


class GRPOTrainer:
    """Trains a policy using GRPO with group-normalized advantages.

    Parameters
    ----------
    policy : nn.Module — the model being trained
    reference : nn.Module — frozen reference policy for KL penalty
    reward_fn : Callable[[str, str], float]
        Takes (problem, completion) and returns a scalar reward.
        For math: 1.0 if correct, 0.0 if wrong.
    config : GRPOConfig
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        reward_fn: Callable[[str, str], float],
        tokenizer,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        for p in self.reference.parameters():
            p.requires_grad = False
        self.loss_history: List[float] = []

    def _sample_completions(self, prompt: str) -> List[str]:
        """Sample G completions from the current policy."""
        self.policy.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        device = next(self.policy.parameters()).device
        input_ids = input_ids.to(device)
        completions = []
        with torch.no_grad():
            for _ in range(self.config.group_size):
                out = self.policy.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                gen_ids = out[0][input_ids.shape[1]:]
                completions.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        self.policy.train()
        return completions

    def train_step(self, problem: str) -> Dict[str, float]:
        """Single GRPO training step for one problem.

        Returns dict of metrics.
        """
        completions = self._sample_completions(problem)
        rewards = torch.tensor(
            [self.reward_fn(problem, c) for c in completions],
            dtype=torch.float32,
        )
        advantages = compute_group_advantages(rewards, self.config.advantage_epsilon)

        # (Simplified) compute scalar log-probs — full implementation tokenizes completions
        # Here we use a stub; see Notebook 04 for full implementation
        policy_log_probs = torch.zeros(self.config.group_size, requires_grad=True)
        old_log_probs = torch.zeros(self.config.group_size)
        ref_log_probs = torch.zeros(self.config.group_size)

        loss, metrics = grpo_loss(
            policy_log_probs, old_log_probs, advantages,
            ref_log_probs=ref_log_probs,
            epsilon_clip=self.config.epsilon_clip,
            kl_coeff=self.config.kl_coeff,
        )
        metrics["rewards_mean"] = rewards.mean().item()
        return metrics


if __name__ == "__main__":
    # Demo: show group advantage normalization
    import numpy as np
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    advantages = compute_group_advantages(rewards)
    print("Rewards:    ", rewards.tolist())
    print("Advantages: ", [round(a, 3) for a in advantages.tolist()])
    print("Mean advantage:", round(advantages.mean().item(), 6), "(should be ≈0)")
    print("Std  advantage:", round(advantages.std().item(), 6), "(should be ≈1)")
