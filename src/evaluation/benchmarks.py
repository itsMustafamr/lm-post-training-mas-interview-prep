"""
benchmarks.py — GSM8K & MATH Benchmark Loaders and Evaluators
==============================================================
Provides synthetic subsets of GSM8K and MATH for demo purposes,
plus answer extraction and evaluation utilities.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic GSM8K sample (20 representative grade-school math problems)
# ─────────────────────────────────────────────────────────────────────────────

GSM8K_SAMPLE: List[Dict[str, object]] = [
    {"question": "A store has 45 apples. It sells 18. How many apples remain?", "answer": 27},
    {"question": "A train travels at 60 km/h for 3 hours. How far does it travel?", "answer": 180},
    {"question": "James has $50. He buys a book for $13 and a pen for $4. How much does he have left?", "answer": 33},
    {"question": "A rectangle is 8 m wide and 5 m tall. What is its area?", "answer": 40},
    {"question": "There are 6 boxes with 12 apples each. How many apples in total?", "answer": 72},
    {"question": "Lisa runs 4 km every day for 7 days. How many km does she run in total?", "answer": 28},
    {"question": "A pizza is cut into 8 slices. 3 slices are eaten. How many slices remain?", "answer": 5},
    {"question": "A car uses 8 liters of fuel per 100 km. How much fuel is needed for 250 km?", "answer": 20},
    {"question": "A school has 24 classrooms with 30 students each. How many students total?", "answer": 720},
    {"question": "Sarah earns $15 per hour and works 8 hours. How much does she earn?", "answer": 120},
    {"question": "A bucket holds 12 liters. It is 3/4 full. How many liters are in it?", "answer": 9},
    {"question": "Tom has 3 bags with 7 marbles each. He gives away 5. How many does he have?", "answer": 16},
    {"question": "A factory produces 150 units per day. How many units in 5 days?", "answer": 750},
    {"question": "A rope 100 m long is cut into 4 equal pieces. How long is each piece?", "answer": 25},
    {"question": "A tank has 200 liters. 75 liters are used. How many liters remain?", "answer": 125},
    {"question": "A bakery makes 36 cookies and packs them 6 per box. How many boxes?", "answer": 6},
    {"question": "A cyclist rides 25 km in the morning and 15 km in the afternoon. Total km?", "answer": 40},
    {"question": "Mark has 5 packs of stickers, 8 per pack. He buys 10 more. Total stickers?", "answer": 50},
    {"question": "A swimming pool is 50 m long and 25 m wide. What is its surface area?", "answer": 1250},
    {"question": "A car travels 360 km in 4 hours. What is its average speed in km/h?", "answer": 90},
]

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic MATH sample (20 competition math problems across difficulty levels)
# ─────────────────────────────────────────────────────────────────────────────

MATH_SAMPLE: List[Dict[str, object]] = [
    {"problem": "Simplify: (3x + 2)(x - 4)", "answer": "3x^2 - 10x - 8",
     "subject": "Algebra", "level": 1},
    {"problem": "Find the roots of x^2 - 5x + 6 = 0", "answer": "x=2 or x=3",
     "subject": "Algebra", "level": 1},
    {"problem": "What is the sum of the first 10 natural numbers?", "answer": 55,
     "subject": "Number Theory", "level": 1},
    {"problem": "Compute: log_2(64)", "answer": 6,
     "subject": "Algebra", "level": 2},
    {"problem": "A triangle has sides 3, 4, 5. What is its area?", "answer": 6,
     "subject": "Geometry", "level": 2},
    {"problem": "Find the derivative of f(x) = x^3 - 2x + 1", "answer": "3x^2 - 2",
     "subject": "Calculus", "level": 2},
    {"problem": "How many ways can 5 people sit in a row?", "answer": 120,
     "subject": "Combinatorics", "level": 2},
    {"problem": "Find the sum: 1 + 2 + 4 + 8 + ... + 2^9", "answer": 1023,
     "subject": "Number Theory", "level": 2},
    {"problem": "Solve: 2^x = 32", "answer": 5,
     "subject": "Algebra", "level": 3},
    {"problem": "What is the probability of rolling two sixes with two fair dice?", "answer": "1/36",
     "subject": "Probability", "level": 3},
    {"problem": "Find the value of sin(π/6)", "answer": "0.5",
     "subject": "Geometry", "level": 3},
    {"problem": "How many primes are less than 30?", "answer": 10,
     "subject": "Number Theory", "level": 3},
    {"problem": "Compute the integral of 3x^2 from 0 to 2", "answer": 8,
     "subject": "Calculus", "level": 3},
    {"problem": "Find all integer solutions to x^2 + y^2 = 25", "answer": "±3,±4 and ±4,±3 and 0,±5 and ±5,0",
     "subject": "Number Theory", "level": 4},
    {"problem": "The sum of three consecutive integers is 48. Find them.", "answer": "15, 16, 17",
     "subject": "Algebra", "level": 4},
    {"problem": "Prove that there are infinitely many primes (Euclid).", "answer": "proof",
     "subject": "Number Theory", "level": 5},
    {"problem": "Find the eigenvalues of [[2,1],[1,2]]", "answer": "1 and 3",
     "subject": "Algebra", "level": 4},
    {"problem": "Compute: C(10, 3)", "answer": 120,
     "subject": "Combinatorics", "level": 2},
    {"problem": "What is 15! mod 100?", "answer": 0,
     "subject": "Number Theory", "level": 4},
    {"problem": "Find the area under y = x^2 from x=0 to x=3", "answer": 9,
     "subject": "Calculus", "level": 3},
]


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_numerical_answer(text: str) -> Optional[float]:
    """Extract a numerical answer from agent response text.

    Looks for patterns like 'The answer is: 27' or 'ANSWER: 27'.

    Parameters
    ----------
    text : str

    Returns
    -------
    float or None
    """
    patterns = [
        r"the answer is[:\s]+(-?\d+(?:\.\d+)?)",
        r"ANSWER[:\s]+(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return float(m.group(1))
    # Fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return None


def score_answer(
    predicted: Optional[float],
    ground_truth: object,
    tolerance: float = 1e-3,
) -> bool:
    """Check if predicted answer matches ground truth.

    Parameters
    ----------
    predicted : float or None
    ground_truth : numeric or str
    tolerance : float
    """
    if predicted is None:
        return False
    try:
        gt = float(str(ground_truth).strip())
        return abs(predicted - gt) <= tolerance
    except (ValueError, TypeError):
        return False


def evaluate_on_benchmark(
    solve_fn,
    benchmark: str = "gsm8k",
    num_problems: Optional[int] = None,
) -> Dict[str, object]:
    """Evaluate a solver function on a benchmark dataset.

    Parameters
    ----------
    solve_fn : callable
        Takes a problem string and returns a response string.
    benchmark : str
        'gsm8k' or 'math'
    num_problems : int, optional
        If given, only evaluate on first N problems.

    Returns
    -------
    dict with keys: accuracy, correct, total, results
    """
    if benchmark.lower() == "gsm8k":
        problems = GSM8K_SAMPLE
        question_key, answer_key = "question", "answer"
    elif benchmark.lower() == "math":
        problems = MATH_SAMPLE
        question_key, answer_key = "problem", "answer"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose 'gsm8k' or 'math'.")

    if num_problems is not None:
        problems = problems[:num_problems]

    results = []
    correct_count = 0

    for prob in problems:
        question = str(prob[question_key])
        gt = prob[answer_key]
        response = solve_fn(question)
        predicted = extract_numerical_answer(response)
        is_correct = score_answer(predicted, gt)
        if is_correct:
            correct_count += 1
        results.append({
            "question": question,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct_count / len(problems) if problems else 0.0
    return {
        "benchmark": benchmark,
        "accuracy": round(accuracy, 4),
        "correct": correct_count,
        "total": len(problems),
        "results": results,
    }


if __name__ == "__main__":
    # Demo: evaluate a mock solver that always says "The answer is: 42"
    def mock_solver(question: str) -> str:
        # Extract first two numbers and add them (very naive)
        nums = re.findall(r"\d+", question)
        if len(nums) >= 2:
            answer = int(nums[0]) - int(nums[1])
        else:
            answer = 42
        return f"The answer is: {answer}"

    result = evaluate_on_benchmark(mock_solver, benchmark="gsm8k", num_problems=5)
    print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
