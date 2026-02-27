r"""
shapley.py — Shapley Value Credit Attribution
==============================================
Theory
------
The Shapley value phi_i(v) gives agent i's "fair share" of the total outcome,
averaged over all possible orderings of agents joining the coalition:

    phi_i(v) = Sum_{S subset N\{i}} [ |S|!(|N|-|S|-1)! / |N|! ] * [v(S+{i}) - v(S)]

Where:
  - N = set of all agents
  - S = coalition not including agent i
  - v(S) = value (outcome quality) when only agents in S participate
  - v(S+{i}) - v(S) = marginal contribution of agent i to coalition S

Properties (Shapley axioms):
  1. Efficiency: Sum phi_i = v(N)   (all credit sums to total outcome)
  2. Symmetry: equal agents get equal credit
  3. Dummy: agents contributing nothing get phi=0
  4. Linearity: phi_i(v+w) = phi_i(v) + phi_i(w)

Reference: Shapley (1953); Yang et al. arXiv:2511.10687
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Dict, List, Optional, Tuple


CoalitionValueFn = Callable[[frozenset], float]


def exact_shapley_values(
    agents: List[str],
    value_fn: CoalitionValueFn,
) -> Dict[str, float]:
    """Compute exact Shapley values by iterating over all 2^N coalitions.

    ⚠️ Exponential in N — only tractable for N ≤ ~10 agents.

    Parameters
    ----------
    agents : list[str]
        Agent identifiers.
    value_fn : callable
        Maps a frozenset of agent IDs to a scalar outcome value.
        Must satisfy v(∅) = 0.

    Returns
    -------
    dict mapping agent_id → Shapley value
    """
    n = len(agents)
    shapley: Dict[str, float] = {a: 0.0 for a in agents}

    for agent in agents:
        others = [a for a in agents if a != agent]
        # Iterate over all subsets S of the other agents
        for size in range(len(others) + 1):
            for coalition_tuple in itertools.combinations(others, size):
                S = frozenset(coalition_tuple)
                S_with_i = S | {agent}

                # Shapley weight: |S|!(|N|-|S|-1)! / |N|!
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                marginal = value_fn(S_with_i) - value_fn(S)
                shapley[agent] += weight * marginal

    return shapley


def approximate_shapley_values(
    agents: List[str],
    value_fn: CoalitionValueFn,
    num_samples: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Monte Carlo approximation of Shapley values.

    Samples random orderings and computes each agent's marginal contribution
    in that ordering. Converges to exact values as num_samples → ∞.

    Parameters
    ----------
    agents : list[str]
    value_fn : callable
    num_samples : int
        Number of random permutations to sample.
    seed : int

    Returns
    -------
    dict mapping agent_id → approximate Shapley value
    """
    rng = random.Random(seed)
    n = len(agents)
    totals: Dict[str, float] = {a: 0.0 for a in agents}

    for _ in range(num_samples):
        order = list(agents)
        rng.shuffle(order)

        coalition: frozenset = frozenset()
        prev_value = value_fn(coalition)

        for agent in order:
            new_coalition = coalition | {agent}
            new_value = value_fn(new_coalition)
            totals[agent] += new_value - prev_value
            prev_value = new_value
            coalition = new_coalition

    return {a: v / num_samples for a, v in totals.items()}


def shapley_to_training_weights(
    shapley_values: Dict[str, float],
    normalize: bool = True,
    min_weight: float = 0.0,
) -> Dict[str, float]:
    """Convert Shapley values to training weight multipliers.

    Positive Shapley → agent contributed; give higher weight to its data.
    Negative Shapley → agent hurt outcome; give lower (or zero) weight.

    Parameters
    ----------
    shapley_values : dict
    normalize : bool — if True, weights sum to len(agents)
    min_weight : float — floor for weights (default 0 — no negative training)
    """
    # Shift so minimum is min_weight
    min_sv = min(shapley_values.values())
    shifted = {a: max(min_weight, v - min_sv) for a, v in shapley_values.items()}

    if normalize:
        total = sum(shifted.values()) or 1.0
        n = len(shifted)
        return {a: v / total * n for a, v in shifted.items()}
    return shifted


class ShapleyCalculator:
    """Convenience class combining exact and approximate Shapley computation.

    Parameters
    ----------
    agents : list[str]
    value_fn : CoalitionValueFn
    use_approximate : bool
        If True (or N > 8), use Monte Carlo approximation.
    num_samples : int
        Samples for MC approximation.
    """

    def __init__(
        self,
        agents: List[str],
        value_fn: CoalitionValueFn,
        use_approximate: bool = False,
        num_samples: int = 200,
        seed: int = 42,
    ) -> None:
        self.agents = agents
        self.value_fn = value_fn
        self.use_approximate = use_approximate or len(agents) > 8
        self.num_samples = num_samples
        self.seed = seed

    def compute(self) -> Dict[str, float]:
        """Compute Shapley values."""
        if self.use_approximate:
            return approximate_shapley_values(
                self.agents, self.value_fn, self.num_samples, self.seed
            )
        return exact_shapley_values(self.agents, self.value_fn)

    def compute_training_weights(self, normalize: bool = True) -> Dict[str, float]:
        """Compute Shapley values and convert to training weights."""
        sv = self.compute()
        return shapley_to_training_weights(sv, normalize=normalize)


if __name__ == "__main__":
    # 3-agent example: solver, critic, reviser
    # v(S) = 1.0 if solver is in S and at least one other, else 0.5 if just solver, else 0.0
    def demo_value_fn(S: frozenset) -> float:
        if "solver" in S and len(S) >= 2:
            return 1.0
        if "solver" in S:
            return 0.5
        return 0.0

    agents = ["solver", "critic", "reviser"]
    calc = ShapleyCalculator(agents, demo_value_fn)
    sv = calc.compute()
    print("Shapley values:", {k: round(v, 4) for k, v in sv.items()})
    weights = calc.compute_training_weights()
    print("Training weights:", {k: round(v, 4) for k, v in weights.items()})
