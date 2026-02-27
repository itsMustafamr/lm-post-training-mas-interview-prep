"""
preference_data.py — Preference Pair Generation for DPO/RLHF
=============================================================
Generates (prompt, chosen, rejected) preference pairs for training
DPO and reward models.

Sources of preference data:
  1. Synthetic: rule-based — correct answer is "chosen", wrong is "rejected"
  2. From MAS traces: agent-scored pairs where better solutions are "chosen"
  3. Random contrast: same question, different quality responses
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from torch.utils.data import Dataset


# Pre-built synthetic preference pairs for demo
SYNTHETIC_PREFERENCES: List[Dict[str, str]] = [
    {
        "prompt": "Solve: A store has 45 apples. It sells 18. How many remain?\n\n### Response:\n",
        "chosen": "Step 1: Start with 45 apples.\nStep 2: Subtract 18 sold: 45 - 18 = 27.\nThe answer is: 27",
        "rejected": "45 + 18 = 63. The answer is: 63",
    },
    {
        "prompt": "Solve: A train travels 60 km/h for 3 hours. How far?\n\n### Response:\n",
        "chosen": "Distance = speed × time = 60 × 3 = 180 km.\nThe answer is: 180",
        "rejected": "The train travels for 3 hours at 60 km/h. Maybe 60 km. The answer is: 60",
    },
    {
        "prompt": "Solve: 6 boxes with 12 apples each. Total apples?\n\n### Response:\n",
        "chosen": "Total = 6 × 12 = 72 apples.\nThe answer is: 72",
        "rejected": "6 + 12 = 18 apples total. The answer is: 18",
    },
    {
        "prompt": "Solve: Rectangle 8m wide and 5m tall. Area?\n\n### Response:\n",
        "chosen": "Area = width × height = 8 × 5 = 40 m².\nThe answer is: 40",
        "rejected": "Area = 8 + 5 = 13 m². The answer is: 13",
    },
    {
        "prompt": "Solve: Sarah earns $15/hour and works 8 hours. Earnings?\n\n### Response:\n",
        "chosen": "Earnings = rate × hours = 15 × 8 = $120.\nThe answer is: 120",
        "rejected": "She earns $15 + $8 = $23. The answer is: 23",
    },
    {
        "prompt": "Solve: Lisa runs 4 km/day for 7 days. Total km?\n\n### Response:\n",
        "chosen": "Total = 4 × 7 = 28 km.\nThe answer is: 28",
        "rejected": "4 km for 7 days, so maybe 4 + 7 = 11 km. The answer is: 11",
    },
    {
        "prompt": "Solve: Car uses 8 liters/100 km. Fuel needed for 250 km?\n\n### Response:\n",
        "chosen": "Fuel = (8/100) × 250 = 20 liters.\nThe answer is: 20",
        "rejected": "250 km ÷ 8 = 31.25 liters. The answer is: 31.25",
    },
    {
        "prompt": "Solve: A rope 100m cut into 4 equal pieces. Length of each?\n\n### Response:\n",
        "chosen": "Each piece = 100 ÷ 4 = 25 m.\nThe answer is: 25",
        "rejected": "100 - 4 = 96 m each. The answer is: 96",
    },
]


def create_synthetic_preferences(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Return a list of synthetic preference pairs.

    Parameters
    ----------
    n : int, optional — number of pairs (if None, return all)
    seed : int — for reproducible shuffling

    Returns
    -------
    list of dicts with keys: 'prompt', 'chosen', 'rejected'
    """
    data = list(SYNTHETIC_PREFERENCES)
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:n] if n is not None else data


def generate_preference_pairs_from_traces(
    traces: List[Dict],
    reward_key: str = "correct",
) -> List[Dict[str, str]]:
    """Generate preference pairs from multi-agent pipeline traces.

    For each problem, the trace with a correct final answer becomes "chosen"
    and traces with incorrect answers become "rejected".

    Parameters
    ----------
    traces : list[dict]
        Each dict should have: 'problem', 'final_solution', and ``reward_key``.
    reward_key : str
        Key used to determine quality ('correct', 'reward', etc.)

    Returns
    -------
    list of dicts with keys: 'prompt', 'chosen', 'rejected'
    """
    # Group traces by problem
    by_problem: Dict[str, List[Dict]] = {}
    for trace in traces:
        problem = trace.get("problem", "")
        by_problem.setdefault(problem, []).append(trace)

    pairs: List[Dict[str, str]] = []
    for problem, problem_traces in by_problem.items():
        chosen_traces = [t for t in problem_traces if t.get(reward_key)]
        rejected_traces = [t for t in problem_traces if not t.get(reward_key)]

        if not chosen_traces or not rejected_traces:
            continue

        # Pair each chosen with each rejected (or just one each for simplicity)
        chosen = chosen_traces[0]
        rejected = rejected_traces[0]
        pairs.append({
            "prompt": f"Solve: {problem}\n\n### Response:\n",
            "chosen": chosen.get("final_solution", ""),
            "rejected": rejected.get("final_solution", ""),
        })

    return pairs


class PreferenceDataset(Dataset):
    """PyTorch Dataset wrapping preference pairs for DPO/RLHF training.

    Parameters
    ----------
    pairs : list[dict] with keys 'prompt', 'chosen', 'rejected'
    """

    def __init__(self, pairs: List[Dict[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.pairs[idx]


if __name__ == "__main__":
    pairs = create_synthetic_preferences(n=3)
    print(f"Created {len(pairs)} preference pairs")
    for i, pair in enumerate(pairs):
        print(f"\n--- Pair {i+1} ---")
        print(f"Prompt:   {pair['prompt'][:60]}...")
        print(f"Chosen:   {pair['chosen'][:60]}...")
        print(f"Rejected: {pair['rejected'][:60]}...")
