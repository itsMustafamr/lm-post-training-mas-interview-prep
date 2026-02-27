"""
rlhf_trainer.py — Simplified RLHF/PPO Trainer for LLM Alignment
=================================================================
Theory
------
RLHF (Reinforcement Learning from Human Feedback) fine-tunes a language
model using PPO (Proximal Policy Optimization).

The three stages:
  Stage 1 — SFT:   π_SFT = argmax_θ L_SFT(θ)
  Stage 2 — RM:    R_φ  trained on human preference pairs
  Stage 3 — PPO:   π_RLHF = argmax_π E[R_φ(x,y)] - β KL(π||π_SFT)

PPO objective (with KL penalty):
    L_PPO = E[ min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t) ] - β KL(π||π_ref)

Where r_t = π_θ(a_t|s_t) / π_old(a_t|s_t) is the probability ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RLHFConfig:
    """Configuration for RLHF/PPO training.

    Attributes
    ----------
    kl_coeff : float
        β — KL divergence penalty from reference policy.
    gamma : float
        Discount factor for returns (usually 1.0 for text).
    gae_lambda : float
        GAE smoothing parameter.
    epsilon_clip : float
        PPO clipping range.
    ppo_epochs : int
        Gradient steps per batch of experience.
    learning_rate : float
    """

    kl_coeff: float = 0.05
    gamma: float = 1.0
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.2
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_new_tokens: int = 256
    logging_steps: int = 10
    seed: int = 42


class PPOBuffer:
    """Stores trajectories (prompt, completion, reward, log_prob) for PPO updates."""

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.completions: List[str] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []

    def add(self, prompt: str, completion: str, reward: float, log_prob: float) -> None:
        self.prompts.append(prompt)
        self.completions.append(completion)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.prompts.clear()
        self.completions.clear()
        self.rewards.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_advantages(
    rewards: List[float],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimates (GAE).

    For the simplified text generation setting where each generation
    is a single step, GAE reduces to just the reward (no value function).

    A_t = R_t - V(s_t)  (simplified: A_t = R_t - mean(R))

    Returns
    -------
    torch.Tensor of shape (N,) — normalized advantages
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    adv = (r - r.mean()) / (r.std() + 1e-8)
    return adv


class RLHFTrainer:
    """Simplified RLHF/PPO trainer for LLM alignment.

    This implements the core PPO loop without a value network
    (appropriate for the single-step text generation setting).

    Parameters
    ----------
    policy : nn.Module — model being trained
    reference : nn.Module — frozen SFT model for KL penalty
    reward_model : callable — takes (prompt, completion) → float
    tokenizer
    config : RLHFConfig
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        reward_model,
        tokenizer,
        config: Optional[RLHFConfig] = None,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config or RLHFConfig()
        self.buffer = PPOBuffer()
        for p in self.reference.parameters():
            p.requires_grad = False
        self.loss_history: List[float] = []

    def collect_experience(self, prompts: List[str]) -> None:
        """Generate completions, score with reward model, store in buffer."""
        self.policy.eval()
        device = next(self.policy.parameters()).device

        for prompt in prompts:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128
            ).input_ids.to(device)

            with torch.no_grad():
                out = self.policy.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][input_ids.shape[1]:]
            completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            reward = float(self.reward_model(prompt, completion))

            # Simple log-prob estimate (stub; full impl computes token-level)
            log_prob = 0.0
            self.buffer.add(prompt, completion, reward, log_prob)

    def ppo_update(self) -> float:
        """Run PPO update on stored buffer. Returns mean loss."""
        rewards = self.buffer.rewards
        advantages = compute_advantages(
            rewards, self.config.gamma, self.config.gae_lambda
        )
        # In a full implementation: compute token-level log-probs and apply clipped surrogate
        # Stub returns the negative mean reward as a proxy loss
        loss_val = -advantages.mean().item()
        self.buffer.clear()
        return loss_val

    def train(self, prompts: List[str], num_iterations: int = 10) -> List[float]:
        """Run RLHF training loop.

        Parameters
        ----------
        prompts : list[str] — training prompts
        num_iterations : int — collect-update cycles
        """
        losses: List[float] = []
        for it in range(num_iterations):
            self.collect_experience(prompts)
            loss = self.ppo_update()
            losses.append(loss)
            if (it + 1) % self.config.logging_steps == 0:
                print(f"  Iteration {it+1} | proxy_loss={loss:.4f} "
                      f"| mean_reward={sum(self.buffer.rewards or [0])/max(len(self.buffer),1):.3f}")
        self.loss_history = losses
        return losses


if __name__ == "__main__":
    print("RLHFTrainer — see Notebook 02 for a full runnable PPO example.")
    print("compute_advantages() demo:")
    rewards = [0.1, 0.9, 0.3, 0.7, 0.5]
    adv = compute_advantages(rewards)
    print("  Rewards:   ", rewards)
    print("  Advantages:", [round(a, 3) for a in adv.tolist()])
