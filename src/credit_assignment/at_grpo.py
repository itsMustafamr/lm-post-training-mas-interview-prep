"""
at_grpo.py — Agent- and Turn-wise GRPO (AT-GRPO)
==================================================
Theory
------
Standard GRPO operates at the *system* level: it samples G completions from
a single agent. AT-GRPO extends this to multi-agent settings by computing
advantages at two granularities:

  1. Agent-level advantage A_agent:
       Normalize rewards across agents on the same problem.
       A_i^agent = (R_i - mean_j(R_j)) / std_j(R_j)
       (Which agent contributed most relative to the group?)

  2. Turn-level advantage A_turn:
       Within a single agent's multi-turn interaction, normalize across turns.
       A_{i,t}^turn = (r_{i,t} - mean_t(r_{i,t})) / std_t(r_{i,t})
       (Which turn within this agent's trajectory was most valuable?)

Combined advantage:
       A_{i,t} = α · A_i^agent + (1-α) · A_{i,t}^turn

Reference: Yang et al. arXiv:2511.10687 §4 AT-GRPO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ATGRPOConfig:
    """Configuration for AT-GRPO.

    Attributes
    ----------
    agent_weight : float
        α — weight for agent-level advantage (vs turn-level).
    epsilon_clip : float
        PPO clipping parameter.
    kl_coeff : float
        KL penalty coefficient.
    advantage_epsilon : float
        Numerical stability constant.
    learning_rate : float
    """

    agent_weight: float = 0.5
    epsilon_clip: float = 0.2
    kl_coeff: float = 0.04
    advantage_epsilon: float = 1e-8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    seed: int = 42


def compute_agent_advantages(
    agent_rewards: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards *across agents* on the same problem.

    Parameters
    ----------
    agent_rewards : (N_agents,)
        Scalar outcome reward for each agent on a given problem.

    Returns
    -------
    torch.Tensor of shape (N_agents,)
    """
    mean = agent_rewards.mean()
    std = agent_rewards.std()
    return (agent_rewards - mean) / (std + epsilon)


def compute_turn_advantages(
    turn_rewards: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards *across turns* within a single agent.

    Parameters
    ----------
    turn_rewards : (N_turns,)
        Step-level reward for each turn of a single agent's trajectory.

    Returns
    -------
    torch.Tensor of shape (N_turns,)
    """
    mean = turn_rewards.mean()
    std = turn_rewards.std()
    return (turn_rewards - mean) / (std + epsilon)


def combine_advantages(
    agent_adv: torch.Tensor,
    turn_adv: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combine agent-level and turn-level advantages.

    A_combined = α · A_agent + (1-α) · A_turn

    Parameters
    ----------
    agent_adv : (N_turns,) — agent-level advantage broadcast to each turn
    turn_adv : (N_turns,)
    alpha : float — weight for agent-level component

    Returns
    -------
    torch.Tensor of shape (N_turns,)
    """
    return alpha * agent_adv + (1.0 - alpha) * turn_adv


def at_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    epsilon_clip: float = 0.2,
    kl_coeff: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """AT-GRPO clipped surrogate loss.

    Same as standard GRPO loss but applied with combined advantages.
    """
    import torch.nn.functional as F

    log_ratio = policy_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantages
    surrogate_loss = -torch.min(surr1, surr2).mean()

    kl_loss = torch.tensor(0.0)
    if ref_log_probs is not None:
        kl_loss = (policy_log_probs - ref_log_probs.detach()).mean()

    total_loss = surrogate_loss + kl_coeff * kl_loss
    metrics = {
        "surrogate_loss": surrogate_loss.item(),
        "kl_loss": kl_loss.item(),
        "mean_advantage": advantages.mean().item(),
    }
    return total_loss, metrics


class ATGRPOTrainer:
    """Multi-agent extension of GRPO using agent- and turn-wise advantages.

    Parameters
    ----------
    agents : list — agent objects with a ``role`` attribute
    config : ATGRPOConfig
    """

    def __init__(self, agents: list, config: Optional[ATGRPOConfig] = None) -> None:
        self.agents = agents
        self.config = config or ATGRPOConfig()

    def compute_combined_advantages(
        self,
        agent_outcome_rewards: Dict[str, float],
        agent_turn_rewards: Dict[str, List[float]],
    ) -> Dict[str, torch.Tensor]:
        """Compute combined AT-GRPO advantages for all agents.

        Parameters
        ----------
        agent_outcome_rewards : dict  agent_id → scalar outcome reward
        agent_turn_rewards : dict  agent_id → list of per-turn rewards

        Returns
        -------
        dict  agent_id → tensor of shape (N_turns,)
        """
        agent_ids = list(agent_outcome_rewards.keys())
        outcome_tensor = torch.tensor(
            [agent_outcome_rewards[a] for a in agent_ids], dtype=torch.float32
        )
        agent_adv_scalars = compute_agent_advantages(
            outcome_tensor, self.config.advantage_epsilon
        )
        agent_adv_map = dict(zip(agent_ids, agent_adv_scalars.tolist()))

        combined: Dict[str, torch.Tensor] = {}
        for agent_id in agent_ids:
            turns = agent_turn_rewards.get(agent_id, [agent_outcome_rewards[agent_id]])
            turn_tensor = torch.tensor(turns, dtype=torch.float32)
            turn_adv = compute_turn_advantages(turn_tensor, self.config.advantage_epsilon)

            # Broadcast agent-level scalar to match turn count
            agent_scalar = torch.full_like(turn_adv, agent_adv_map[agent_id])
            combined[agent_id] = combine_advantages(
                agent_scalar, turn_adv, alpha=self.config.agent_weight
            )

        return combined


if __name__ == "__main__":
    # Demo: 3 agents, 3 turns each
    agent_rewards = {"solver": 0.3, "critic": 0.8, "reviser": 1.0}
    turn_rewards = {
        "solver":  [0.1, 0.2, 0.3],
        "critic":  [0.5, 0.9, 0.8],
        "reviser": [0.7, 0.9, 1.0],
    }
    trainer = ATGRPOTrainer(agents=[], config=ATGRPOConfig(agent_weight=0.5))
    advantages = trainer.compute_combined_advantages(agent_rewards, turn_rewards)
    print("AT-GRPO combined advantages:")
    for agent, adv in advantages.items():
        print(f"  {agent}: {[round(a, 3) for a in adv.tolist()]}")
