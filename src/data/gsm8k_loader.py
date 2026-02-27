"""
gsm8k_loader.py — GSM8K Dataset Loader
========================================
GSM8K (Grade School Math 8K) is a dataset of 8,500 linguistically diverse
grade school math word problems. This module provides a synthetic subset for
demos and a loader class for working with the dataset.

Reference: Cobbe et al. "Training Verifiers to Solve Math Word Problems" (2021)
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

# Full synthetic sample (same as benchmarks.py for consistency)
GSM8K_PROBLEMS: List[Dict[str, object]] = [
    {"question": "A store has 45 apples. It sells 18. How many apples remain?", "answer": 27},
    {"question": "A train travels at 60 km/h for 3 hours. How far does it travel?", "answer": 180},
    {"question": "James has $50. He buys a book for $13 and a pen for $4. How much left?", "answer": 33},
    {"question": "A rectangle is 8 m wide and 5 m tall. What is its area?", "answer": 40},
    {"question": "There are 6 boxes with 12 apples each. How many apples in total?", "answer": 72},
    {"question": "Lisa runs 4 km every day for 7 days. How many km total?", "answer": 28},
    {"question": "A pizza is cut into 8 slices. 3 slices are eaten. How many remain?", "answer": 5},
    {"question": "A car uses 8 liters per 100 km. How much fuel for 250 km?", "answer": 20},
    {"question": "A school has 24 classrooms with 30 students each. Total students?", "answer": 720},
    {"question": "Sarah earns $15/hour and works 8 hours. How much does she earn?", "answer": 120},
    {"question": "A bucket holds 12 liters. It is 3/4 full. How many liters?", "answer": 9},
    {"question": "Tom has 3 bags with 7 marbles each. He gives away 5. How many remain?", "answer": 16},
    {"question": "A factory produces 150 units per day. Units in 5 days?", "answer": 750},
    {"question": "A rope 100 m long is cut into 4 equal pieces. Length of each?", "answer": 25},
    {"question": "A tank has 200 liters. 75 liters are used. How many remain?", "answer": 125},
    {"question": "A bakery makes 36 cookies, packs 6 per box. How many boxes?", "answer": 6},
    {"question": "Cyclist rides 25 km morning and 15 km afternoon. Total km?", "answer": 40},
    {"question": "Mark has 5 packs of stickers, 8 per pack. Buys 10 more. Total?", "answer": 50},
    {"question": "A pool is 50 m long and 25 m wide. Surface area?", "answer": 1250},
    {"question": "A car travels 360 km in 4 hours. Average speed in km/h?", "answer": 90},
]


class GSM8KLoader:
    """Loader for the synthetic GSM8K dataset.

    Parameters
    ----------
    problems : list[dict], optional
        Custom problem list. Defaults to the built-in GSM8K_PROBLEMS.
    shuffle : bool
        If True, shuffle the dataset on load.
    seed : int
    """

    def __init__(
        self,
        problems: Optional[List[Dict]] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self._data = list(problems or GSM8K_PROBLEMS)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def load(self) -> List[Dict[str, object]]:
        """Return the full problem list."""
        return list(self._data)

    def get_batch(self, start: int = 0, size: int = 10) -> List[Dict[str, object]]:
        """Return a contiguous batch of problems.

        Parameters
        ----------
        start : int — starting index
        size : int  — batch size
        """
        return self._data[start: start + size]

    def iterate(self, batch_size: int = 1) -> Iterator[List[Dict[str, object]]]:
        """Iterate over the dataset in batches.

        Yields
        ------
        list[dict] of length ``batch_size``
        """
        for i in range(0, len(self._data), batch_size):
            yield self._data[i: i + batch_size]

    def get_questions(self) -> List[str]:
        """Return just the question strings."""
        return [d["question"] for d in self._data]

    def get_answers(self) -> List[object]:
        """Return just the answer values."""
        return [d["answer"] for d in self._data]


if __name__ == "__main__":
    loader = GSM8KLoader(shuffle=True)
    print(f"Loaded {len(loader)} GSM8K problems")
    for batch in loader.iterate(batch_size=3):
        for item in batch:
            print(f"  Q: {item['question'][:60]}... A: {item['answer']}")
        break  # Show first batch only
