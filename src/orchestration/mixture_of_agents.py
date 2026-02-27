"""
mixture_of_agents.py — Mixture-of-Agents (MoA) Orchestrator
=============================================================
Based on "Mixture-of-Agents Enhances Large Language Model Capabilities"
(Wang et al., 2024).

Architecture:
  Layer 1 (Proposers): N agents independently generate diverse solutions.
  Layer 2 (Aggregator): 1 agent synthesizes all Layer-1 responses into
                        a single, higher-quality answer.

The key insight is that aggregating multiple independent views consistently
outperforms any individual model, even when all models are the same size.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..agents.base_agent import BaseAgent
from .logger import TraceLogger


class MixtureOfAgents:
    """Two-layer MoA: diverse proposers + single aggregator.

    Parameters
    ----------
    proposers : list[BaseAgent]
        Agents in the first (generation) layer. Should be at least 2.
    aggregator : BaseAgent
        Agent in the second (aggregation) layer.
    logger : TraceLogger, optional
    """

    def __init__(
        self,
        proposers: List[BaseAgent],
        aggregator: BaseAgent,
        logger: Optional[TraceLogger] = None,
    ) -> None:
        if len(proposers) < 1:
            raise ValueError("MoA requires at least one proposer.")
        self.proposers = proposers
        self.aggregator = aggregator
        self.logger = logger or TraceLogger()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, problem: str, ground_truth: Optional[float] = None) -> Dict:
        """Run MoA on a single problem.

        Returns
        -------
        dict
            Keys: problem, proposer_responses, aggregated_response,
                  final_answer, correct, trace_summary.
        """
        self.logger.reset()
        for agent in self.proposers + [self.aggregator]:
            agent.reset()

        # ── Layer 1: Independent diverse generation ────────────────────
        proposer_responses: List[str] = []
        for idx, agent in enumerate(self.proposers):
            result = agent.call(problem)
            proposer_responses.append(result["response"])
            self.logger.log_message(
                agent.agent_id, agent.role, result["response"],
                round_idx=0, step_idx=idx,
                extra={"layer": "proposer", "latency_ms": result["latency_ms"]},
            )

        # ── Layer 2: Aggregation ───────────────────────────────────────
        proposals_text = "\n\n".join(
            f"Proposal {i+1}:\n{r}" for i, r in enumerate(proposer_responses)
        )
        agg_prompt = (
            f"Problem: {problem}\n\n"
            f"The following {len(proposer_responses)} agents independently solved "
            f"this problem:\n\n{proposals_text}\n\n"
            "Synthesize the best elements of each proposal into a single, "
            "accurate, step-by-step solution. End with 'The answer is: [NUMBER]'."
        )
        agg_result = self.aggregator.call(agg_prompt)
        aggregated = agg_result["response"]
        self.logger.log_message(
            self.aggregator.agent_id, self.aggregator.role, aggregated,
            round_idx=1, step_idx=0,
            extra={"layer": "aggregator", "latency_ms": agg_result["latency_ms"]},
        )

        # Extract final answer
        import re
        match = re.search(r"the answer is:\s*(-?\d+(?:\.\d+)?)", aggregated, re.IGNORECASE)
        final_answer: Optional[float] = float(match.group(1)) if match else None
        correct: Optional[bool] = None
        if ground_truth is not None and final_answer is not None:
            correct = abs(final_answer - ground_truth) < 1e-3

        return {
            "problem": problem,
            "proposer_responses": proposer_responses,
            "aggregated_response": aggregated,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "correct": correct,
            "trace_summary": self.logger.get_summary(),
        }


if __name__ == "__main__":
    import json
    from ..agents.solver_agent import SolverAgent

    proposers = [SolverAgent(agent_id=f"proposer_{i}") for i in range(3)]
    aggregator = SolverAgent(agent_id="aggregator")
    moa = MixtureOfAgents(proposers, aggregator)
    result = moa.run("A rectangle is 8 cm wide and 5 cm tall. What is its area?",
                     ground_truth=40.0)
    print(json.dumps(result, indent=2, default=str))
