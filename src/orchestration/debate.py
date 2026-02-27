"""
debate.py — Multi-Agent Debate Orchestrator
=============================================
Based on "Improving Factuality and Reasoning in Language Models through
Multiagent Debate" (Du et al., 2023).

Each round:
  1. All agents independently generate a response.
  2. Each agent reads *all other agents'* previous responses.
  3. Agents update their answers considering the ensemble.
  4. Convergence is declared when a threshold fraction agree.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..agents.base_agent import BaseAgent
from .logger import TraceLogger


def _majority_answer(responses: List[str]) -> Optional[str]:
    """Find the most common numerical answer across responses."""
    answers: Dict[str, int] = {}
    for r in responses:
        nums = re.findall(r"answer is:\s*(-?\d+(?:\.\d+)?)", r, re.IGNORECASE)
        if nums:
            key = nums[-1]
            answers[key] = answers.get(key, 0) + 1
    if not answers:
        return None
    return max(answers, key=lambda k: answers[k])


class DebateOrchestrator:
    """Runs a multi-agent debate for a fixed number of rounds.

    Parameters
    ----------
    agents : list[BaseAgent]
        Participating agents (should all be of similar capability).
    num_rounds : int
        Maximum debate rounds.
    convergence_threshold : float
        Fraction of agents that must agree to stop early (default 0.67).
    logger : TraceLogger, optional
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        num_rounds: int = 3,
        convergence_threshold: float = 0.67,
        logger: Optional[TraceLogger] = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Debate requires at least 2 agents.")
        self.agents = agents
        self.num_rounds = num_rounds
        self.convergence_threshold = convergence_threshold
        self.logger = logger or TraceLogger()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, problem: str, ground_truth: Optional[float] = None) -> Dict:
        """Run the debate on ``problem``.

        Returns
        -------
        dict
            Keys: problem, final_answer, correct, rounds_to_converge,
                  agreement_per_round, all_responses, trace_summary.
        """
        self.logger.reset()
        for agent in self.agents:
            agent.reset()

        # Initial independent responses (round 0)
        current_responses: List[str] = [""] * len(self.agents)
        agreement_per_round: List[float] = []
        rounds_to_converge = self.num_rounds

        for round_idx in range(self.num_rounds):
            new_responses: List[str] = []

            for agent_idx, agent in enumerate(self.agents):
                if round_idx == 0:
                    prompt = problem
                else:
                    # Show this agent what all *other* agents said last round
                    others = [
                        f"Agent {i+1}: {current_responses[i]}"
                        for i in range(len(self.agents))
                        if i != agent_idx
                    ]
                    others_text = "\n\n".join(others)
                    prompt = (
                        f"Problem: {problem}\n\n"
                        f"Other agents' solutions:\n{others_text}\n\n"
                        f"Your previous answer: {current_responses[agent_idx]}\n\n"
                        "Considering the above, give your updated solution."
                    )

                result = agent.call(prompt)
                new_responses.append(result["response"])
                self.logger.log_message(
                    agent.agent_id, agent.role, result["response"],
                    round_idx=round_idx, step_idx=agent_idx,
                    extra={"latency_ms": result["latency_ms"]},
                )

            current_responses = new_responses

            # ── Convergence check ──────────────────────────────────────
            majority = _majority_answer(current_responses)
            if majority is not None:
                agreeing = sum(
                    1 for r in current_responses
                    if majority in r
                )
                agreement = agreeing / len(self.agents)
            else:
                agreement = 0.0
            agreement_per_round.append(round(agreement, 3))

            if agreement >= self.convergence_threshold:
                rounds_to_converge = round_idx + 1
                break

        # Extract final answer
        majority_ans = _majority_answer(current_responses)
        final_answer: Optional[float] = None
        if majority_ans is not None:
            try:
                final_answer = float(majority_ans)
            except ValueError:
                pass

        correct: Optional[bool] = None
        if ground_truth is not None and final_answer is not None:
            correct = abs(final_answer - ground_truth) < 1e-3

        return {
            "problem": problem,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "correct": correct,
            "rounds_to_converge": rounds_to_converge,
            "agreement_per_round": agreement_per_round,
            "all_final_responses": current_responses,
            "trace_summary": self.logger.get_summary(),
        }


if __name__ == "__main__":
    import json
    from ..agents.solver_agent import SolverAgent

    agents = [SolverAgent(agent_id=f"solver_{i}") for i in range(3)]
    debate = DebateOrchestrator(agents, num_rounds=3)
    result = debate.run("A train travels 60 km/h for 2 hours. How far does it go?",
                        ground_truth=120.0)
    print(json.dumps(result, indent=2, default=str))
