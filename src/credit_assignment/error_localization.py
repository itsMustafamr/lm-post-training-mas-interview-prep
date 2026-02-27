"""
error_localization.py — Error Localization in Agent Traces
===========================================================
When a multi-agent pipeline produces a wrong answer, we need to identify
*which agent* introduced the error — so we can assign blame/credit signals
for targeted re-training.

Algorithm
---------
1. Walk the trace chronologically (step by step).
2. At each step, check whether the current state is "on the right track"
   (partial correctness check or key-number presence).
3. The first step where the state diverges from the correct path is the
   "error introduction point".
4. Assign blame signal (-1) to the agent at that step, credit signal (+1)
   to agents after a correction, and neutral (0) otherwise.

Reference: Yang et al. arXiv:2511.10687 §3.2 Error Localization
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def _extract_numbers(text: str) -> List[float]:
    """Extract all numerical values from text."""
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(x) for x in found]


def _is_on_track(text: str, ground_truth: float, tolerance: float = 0.01) -> bool:
    """Heuristic: is the ground truth answer mentioned anywhere in text?"""
    numbers = _extract_numbers(text)
    return any(abs(n - ground_truth) <= tolerance * max(1, abs(ground_truth))
               for n in numbers)


def locate_first_error(
    trace: List[Dict],
    ground_truth: float,
) -> Optional[int]:
    """Find the index of the first trace entry that introduces an error.

    Parameters
    ----------
    trace : list[dict]
        Trace from TraceLogger — each entry has 'content', 'agent_id', 'role'.
    ground_truth : float
        The correct answer.

    Returns
    -------
    int or None
        Index of the first error entry, or None if no error found.
    """
    for i, entry in enumerate(trace):
        role = entry.get("role", "")
        content = entry.get("content", "")

        # Only check solver/reviser outputs (they produce the answer)
        if role not in ("solver", "reviser"):
            continue

        if not _is_on_track(content, ground_truth):
            return i
    return None


def assign_blame_credit(
    trace: List[Dict],
    ground_truth: float,
    final_correct: bool,
) -> Dict[str, float]:
    """Assign blame/credit signals to each agent in the trace.

    Rules:
      - If final answer is correct and agent contributed: +1
      - If first error was introduced by this agent: -1
      - If agent corrected a prior error: +0.5
      - Otherwise: 0

    Parameters
    ----------
    trace : list[dict]
    ground_truth : float
    final_correct : bool

    Returns
    -------
    dict mapping agent_id → signal value ∈ [-1, +1]
    """
    agents_seen = list({e["agent_id"] for e in trace})
    signals: Dict[str, float] = {a: 0.0 for a in agents_seen}

    if final_correct:
        # Give credit to all agents that contributed
        for entry in trace:
            signals[entry["agent_id"]] = signals.get(entry["agent_id"], 0.0) + 0.1
        # Normalize to max +1
        max_sig = max(signals.values()) or 1.0
        signals = {a: min(1.0, v / max_sig) for a, v in signals.items()}
        return signals

    # Final answer wrong: locate first error
    error_idx = locate_first_error(trace, ground_truth)
    if error_idx is not None:
        blame_agent = trace[error_idx]["agent_id"]
        signals[blame_agent] = -1.0

        # Check if subsequent agents corrected it
        for entry in trace[error_idx + 1:]:
            if _is_on_track(entry.get("content", ""), ground_truth):
                signals[entry["agent_id"]] = max(
                    signals.get(entry["agent_id"], 0.0), 0.5
                )

    return signals


class ErrorLocalizer:
    """Convenience wrapper for error localization and blame assignment.

    Parameters
    ----------
    ground_truth : float
        The correct numerical answer.
    """

    def __init__(self, ground_truth: float) -> None:
        self.ground_truth = ground_truth

    def locate_error(self, trace: List[Dict]) -> Optional[int]:
        """Return index of first error in trace, or None."""
        return locate_first_error(trace, self.ground_truth)

    def assign_signals(
        self, trace: List[Dict], final_correct: bool
    ) -> Dict[str, float]:
        """Return per-agent blame/credit signals."""
        return assign_blame_credit(trace, self.ground_truth, final_correct)

    def get_report(self, trace: List[Dict], final_correct: bool) -> Dict:
        """Full report: error location + signals."""
        error_idx = self.locate_error(trace)
        signals = self.assign_signals(trace, final_correct)
        return {
            "ground_truth": self.ground_truth,
            "final_correct": final_correct,
            "first_error_index": error_idx,
            "first_error_agent": trace[error_idx]["agent_id"] if error_idx is not None else None,
            "agent_signals": signals,
        }


if __name__ == "__main__":
    # Demo trace: solver makes error, reviser fixes it
    trace = [
        {"agent_id": "solver_0", "role": "solver",
         "content": "Step 1: 45 + 18 = 63. The answer is: 63"},  # Wrong (should subtract)
        {"agent_id": "critic_0", "role": "critic",
         "content": "VERDICT: INCORRECT — should subtract, not add"},
        {"agent_id": "reviser_0", "role": "reviser",
         "content": "Step 1: 45 - 18 = 27. The answer is: 27"},  # Correct
    ]
    localizer = ErrorLocalizer(ground_truth=27.0)
    report = localizer.get_report(trace, final_correct=True)
    import json
    print(json.dumps(report, indent=2))
