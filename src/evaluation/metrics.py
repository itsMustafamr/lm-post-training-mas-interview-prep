"""
metrics.py — Evaluation Metrics for MAS and Training
======================================================
Provides metrics for:
  - Individual model evaluation (accuracy, pass@k)
  - Multi-agent system evaluation (convergence rate, error correction rate)
  - Training evaluation (loss curves, reward trends)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def compute_accuracy(
    correct: List[bool],
) -> float:
    """Fraction of correct answers."""
    if not correct:
        return 0.0
    return sum(correct) / len(correct)


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """Compute pass@k metric (Chen et al., HumanEval).

    pass@k = 1 - C(n-c, k) / C(n, k)

    Parameters
    ----------
    n : int — total number of samples per problem
    c : int — number of correct samples
    k : int — number of samples used for evaluation
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_convergence_rate(
    rounds_to_converge: List[int],
    max_rounds: int,
) -> float:
    """Fraction of problems that converged before max_rounds.

    Parameters
    ----------
    rounds_to_converge : list[int]
        Number of rounds each problem took to converge.
        Use max_rounds+1 for problems that never converged.
    max_rounds : int
    """
    if not rounds_to_converge:
        return 0.0
    converged = sum(1 for r in rounds_to_converge if r < max_rounds)
    return converged / len(rounds_to_converge)


def compute_error_correction_rate(
    results: List[Dict],
) -> float:
    """Fraction of initially-wrong problems that were corrected by the pipeline.

    Parameters
    ----------
    results : list[dict]
        Each dict should have 'initial_correct' (bool) and 'final_correct' (bool).
    """
    initially_wrong = [r for r in results if not r.get("initial_correct", True)]
    if not initially_wrong:
        return 0.0
    corrected = sum(1 for r in initially_wrong if r.get("final_correct", False))
    return corrected / len(initially_wrong)


def compute_collaboration_efficiency(
    single_agent_accuracy: float,
    multi_agent_accuracy: float,
    num_agents: int,
) -> float:
    """Measure how efficiently the multi-agent setup uses extra compute.

    Efficiency = (accuracy improvement) / (compute cost increase)
               = (MAS_acc - SA_acc) / (num_agents - 1)

    Returns
    -------
    float — improvement per additional agent (higher is better)
    """
    if num_agents <= 1:
        return 0.0
    improvement = multi_agent_accuracy - single_agent_accuracy
    cost_increase = num_agents - 1
    return improvement / cost_increase


class MetricsTracker:
    """Tracks and aggregates evaluation metrics across runs.

    Usage
    -----
    >>> tracker = MetricsTracker()
    >>> tracker.record(correct=True, rounds=2, initial_correct=False)
    >>> tracker.record(correct=False, rounds=3, initial_correct=False)
    >>> print(tracker.summary())
    """

    def __init__(self) -> None:
        self.records: List[Dict] = []

    def record(
        self,
        correct: bool,
        rounds: int = 1,
        initial_correct: Optional[bool] = None,
    ) -> None:
        """Record a single problem result."""
        self.records.append({
            "correct": correct,
            "rounds": rounds,
            "initial_correct": initial_correct if initial_correct is not None else correct,
        })

    def summary(self, max_rounds: int = 3) -> Dict[str, float]:
        """Compute summary statistics across all recorded results."""
        if not self.records:
            return {}
        correct_flags = [r["correct"] for r in self.records]
        rounds = [r["rounds"] for r in self.records]
        return {
            "accuracy": round(compute_accuracy(correct_flags), 4),
            "mean_rounds": round(sum(rounds) / len(rounds), 2),
            "convergence_rate": round(compute_convergence_rate(rounds, max_rounds), 4),
            "error_correction_rate": round(compute_error_correction_rate(self.records), 4),
            "n_problems": len(self.records),
        }

    def reset(self) -> None:
        """Clear all recorded results."""
        self.records = []


if __name__ == "__main__":
    # Demo
    tracker = MetricsTracker()
    simulated = [
        (True, 1, True), (True, 2, False), (False, 3, False),
        (True, 1, True), (True, 2, False),
    ]
    for correct, rounds, initial in simulated:
        tracker.record(correct=correct, rounds=rounds, initial_correct=initial)

    print("Metrics summary:")
    for k, v in tracker.summary().items():
        print(f"  {k}: {v}")

    # pass@k demo
    print(f"\npass@1 (n=10, c=3): {compute_pass_at_k(10, 3, 1):.3f}")
    print(f"pass@10 (n=10, c=3): {compute_pass_at_k(10, 3, 10):.3f}")
