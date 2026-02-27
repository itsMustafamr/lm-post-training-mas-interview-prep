"""
process_reward.py — Process Reward Model (PRM) for Step-Level Credit
=====================================================================
Theory
------
Outcome Reward Models (ORM) give a single reward at the end of a sequence.
Process Reward Models (PRM) assign rewards to *each reasoning step*, enabling:
  - More precise credit assignment within a single agent's output
  - Better supervision signal for long reasoning chains
  - Early detection of errors before they propagate

PRM formula for step t in a sequence of T steps:
    r_t = PRM(x, y_1, ..., y_t)  ∈ [0, 1]

Training signal: use the final outcome reward (binary) and interpolate
backwards to credit intermediate steps that led to it.

Reference: Lightman et al. "Let's Verify Step by Step" (2023)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def split_into_steps(text: str) -> List[str]:
    """Split a reasoning chain into individual steps.

    Heuristic: split on "Step N:" patterns or newlines.

    Parameters
    ----------
    text : str
        A multi-step reasoning chain.

    Returns
    -------
    list[str] — individual reasoning steps
    """
    # Try "Step N:" pattern first
    steps = re.split(r"(?=Step\s+\d+[:.])", text, flags=re.IGNORECASE)
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) <= 1:
        # Fall back to splitting on newlines
        steps = [s.strip() for s in text.split("\n") if s.strip()]
    return steps if steps else [text]


def interpolate_rewards(
    outcome_reward: float,
    num_steps: int,
    decay: float = 0.9,
) -> List[float]:
    """Create step-level rewards from a single outcome reward using exponential decay.

    Final step gets the full reward; earlier steps get exponentially discounted versions.

    r_t = outcome_reward * decay^(T - t - 1)

    Parameters
    ----------
    outcome_reward : float
        The final outcome reward (e.g., 1.0 for correct, 0.0 for wrong).
    num_steps : int
        Number of reasoning steps.
    decay : float
        Discount per step back from the outcome.

    Returns
    -------
    list[float] of length num_steps
    """
    return [
        outcome_reward * (decay ** (num_steps - t - 1))
        for t in range(num_steps)
    ]


def assign_step_rewards(
    solution_text: str,
    outcome_reward: float,
    decay: float = 0.9,
) -> List[Tuple[str, float]]:
    """Split solution into steps and assign step-level rewards.

    Parameters
    ----------
    solution_text : str
        The full reasoning chain.
    outcome_reward : float
        Binary (or continuous) outcome reward.
    decay : float
        Backward discount factor.

    Returns
    -------
    list of (step_text, step_reward) tuples
    """
    steps = split_into_steps(solution_text)
    rewards = interpolate_rewards(outcome_reward, len(steps), decay)
    return list(zip(steps, rewards))


class StepRewardModel(nn.Module):
    """Neural process reward model: scores each reasoning step.

    Architecture: transformer backbone → linear head per step.
    In the simplified (demo) version, we use heuristics instead of
    a trained network to avoid requiring a GPU.

    Parameters
    ----------
    use_heuristic : bool
        If True, use rule-based heuristics (no GPU needed).
        If False, use the neural backbone.
    """

    def __init__(self, use_heuristic: bool = True) -> None:
        super().__init__()
        self.use_heuristic = use_heuristic
        if not use_heuristic:
            # Placeholder for neural backbone — see Notebook 06 for full impl
            self.backbone = nn.Linear(768, 1)

    def score_step(self, step_text: str, ground_truth: Optional[float] = None) -> float:
        """Score a single reasoning step.

        Returns
        -------
        float in [0, 1]
        """
        if self.use_heuristic:
            return self._heuristic_score(step_text, ground_truth)
        # Neural path (stub)
        return 0.5

    def _heuristic_score(self, step: str, ground_truth: Optional[float]) -> float:
        """Simple heuristic: does this step contain a plausible number?"""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", step)
        if not numbers:
            return 0.3  # Step mentions no numbers — probably low-info
        if ground_truth is not None:
            # Reward higher if ground truth appears in the step
            if any(abs(float(n) - ground_truth) < 1e-3 for n in numbers):
                return 1.0
        return 0.6  # Has numbers, but not the final answer

    def score_solution(
        self,
        solution_text: str,
        ground_truth: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Score all steps in a solution.

        Returns
        -------
        list of (step_text, score) tuples
        """
        steps = split_into_steps(solution_text)
        return [(step, self.score_step(step, ground_truth)) for step in steps]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural forward (only used when use_heuristic=False)."""
        return self.backbone(x)


if __name__ == "__main__":
    solution = (
        "Step 1: The store starts with 45 apples.\n"
        "Step 2: It sells 18 apples. So 45 - 18 = 27.\n"
        "Step 3: The remaining count is 27.\n"
        "The answer is: 27"
    )
    prm = StepRewardModel(use_heuristic=True)
    scored = prm.score_solution(solution, ground_truth=27.0)
    print("Step-level rewards:")
    for step, reward in scored:
        print(f"  [{reward:.2f}] {step[:60]}")

    print("\nInterpolated rewards (outcome=1.0, decay=0.9):")
    for step, reward in assign_step_rewards(solution, outcome_reward=1.0):
        print(f"  [{reward:.3f}] {step[:60]}")
