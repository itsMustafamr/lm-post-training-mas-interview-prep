"""
math_loader.py — MATH Dataset Loader
======================================
The MATH dataset contains 12,500 challenging competition mathematics problems
across 7 subjects and 5 difficulty levels.

Reference: Hendrycks et al. "Measuring Mathematical Problem Solving with MATH" (2021)
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

MATH_PROBLEMS: List[Dict[str, object]] = [
    {"problem": "Simplify: (3x + 2)(x - 4)", "answer": "3x^2 - 10x - 8",
     "subject": "Algebra", "level": 1},
    {"problem": "Find the roots of x^2 - 5x + 6 = 0", "answer": "x=2,x=3",
     "subject": "Algebra", "level": 1},
    {"problem": "Sum of first 10 natural numbers?", "answer": 55,
     "subject": "Number Theory", "level": 1},
    {"problem": "Compute: log_2(64)", "answer": 6,
     "subject": "Algebra", "level": 2},
    {"problem": "A right triangle has legs 3 and 4. What is its area?", "answer": 6,
     "subject": "Geometry", "level": 2},
    {"problem": "Derivative of f(x) = x^3 - 2x + 1", "answer": "3x^2 - 2",
     "subject": "Calculus", "level": 2},
    {"problem": "How many ways can 5 people sit in a row?", "answer": 120,
     "subject": "Combinatorics", "level": 2},
    {"problem": "Sum: 1 + 2 + 4 + 8 + ... + 2^9", "answer": 1023,
     "subject": "Number Theory", "level": 2},
    {"problem": "Solve: 2^x = 32", "answer": 5,
     "subject": "Algebra", "level": 3},
    {"problem": "Probability of rolling two sixes with fair dice?", "answer": "1/36",
     "subject": "Probability", "level": 3},
    {"problem": "Value of sin(π/6)", "answer": 0.5,
     "subject": "Geometry", "level": 3},
    {"problem": "How many primes are less than 30?", "answer": 10,
     "subject": "Number Theory", "level": 3},
    {"problem": "Integral of 3x^2 from 0 to 2", "answer": 8,
     "subject": "Calculus", "level": 3},
    {"problem": "Three consecutive integers sum to 48. Find them.", "answer": "15,16,17",
     "subject": "Algebra", "level": 4},
    {"problem": "Eigenvalues of [[2,1],[1,2]]", "answer": "1 and 3",
     "subject": "Algebra", "level": 4},
    {"problem": "Compute: C(10, 3)", "answer": 120,
     "subject": "Combinatorics", "level": 2},
    {"problem": "What is 15! mod 100?", "answer": 0,
     "subject": "Number Theory", "level": 4},
    {"problem": "Area under y = x^2 from x=0 to x=3", "answer": 9,
     "subject": "Calculus", "level": 3},
    {"problem": "Find the area of a circle with radius 5.", "answer": "25π ≈ 78.54",
     "subject": "Geometry", "level": 2},
    {"problem": "How many subsets does a set of 4 elements have?", "answer": 16,
     "subject": "Combinatorics", "level": 3},
]


class MATHLoader:
    """Loader for the synthetic MATH dataset with filtering support.

    Parameters
    ----------
    problems : list[dict], optional
    subject : str, optional — filter by subject (e.g., 'Algebra')
    max_level : int, optional — only include problems at or below this difficulty
    shuffle : bool
    seed : int
    """

    SUBJECTS = {"Algebra", "Number Theory", "Geometry", "Calculus",
                "Combinatorics", "Probability"}

    def __init__(
        self,
        problems: Optional[List[Dict]] = None,
        subject: Optional[str] = None,
        max_level: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        data = list(problems or MATH_PROBLEMS)
        if subject:
            data = [d for d in data if d.get("subject") == subject]
        if max_level is not None:
            data = [d for d in data if d.get("level", 5) <= max_level]
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(data)
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def load(self) -> List[Dict[str, object]]:
        return list(self._data)

    def get_batch(self, start: int = 0, size: int = 10) -> List[Dict[str, object]]:
        return self._data[start: start + size]

    def iterate(self, batch_size: int = 1) -> Iterator[List[Dict[str, object]]]:
        for i in range(0, len(self._data), batch_size):
            yield self._data[i: i + batch_size]

    def by_level(self, level: int) -> List[Dict[str, object]]:
        """Return problems of exactly this difficulty level."""
        return [d for d in self._data if d.get("level") == level]


if __name__ == "__main__":
    # All problems
    loader = MATHLoader()
    print(f"Total problems: {len(loader)}")

    # Filter by subject
    algebra = MATHLoader(subject="Algebra")
    print(f"Algebra problems: {len(algebra)}")

    # Filter by level
    easy = MATHLoader(max_level=2)
    print(f"Level ≤ 2 problems: {len(easy)}")

    for batch in easy.iterate(batch_size=2):
        for item in batch:
            print(f"  [{item['subject']} L{item['level']}] {item['problem'][:50]} → {item['answer']}")
        break
