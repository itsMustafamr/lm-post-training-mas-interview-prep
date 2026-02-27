"""
pipeline.py — Sequential Multi-Agent Pipeline Orchestrator
============================================================
Runs a list of agents in order: Solver → Critic → Reviser → Verifier.
Supports early stopping when the Verifier confirms a CORRECT answer.

Architecture
------------
  Problem ──► Solver ──► Critic ──► Reviser ──► Verifier ──► Result
               └──────────────────────┘ (repeat up to max_rounds)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..agents.base_agent import BaseAgent
from ..agents.critic_agent import CriticAgent
from ..agents.verifier_agent import VerifierAgent
from .logger import TraceLogger


class PipelineOrchestrator:
    """Runs agents sequentially; repeats critique-revision loop until convergence.

    Parameters
    ----------
    agents : list[BaseAgent]
        Ordered list of agents. Typically [Solver, Critic, Reviser, Verifier].
    max_rounds : int
        Maximum number of solver→critic→reviser cycles before giving up.
    logger : TraceLogger, optional
        If None, a fresh TraceLogger is created per run.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        max_rounds: int = 3,
        logger: Optional[TraceLogger] = None,
    ) -> None:
        self.agents = agents
        self.max_rounds = max_rounds
        self.logger = logger or TraceLogger()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, problem: str, ground_truth: Optional[float] = None) -> Dict:
        """Run the pipeline on a single math problem.

        Parameters
        ----------
        problem : str
            The math problem statement.
        ground_truth : float, optional
            Known correct answer (used by verifier for accuracy tracking).

        Returns
        -------
        dict
            Keys: 'problem', 'final_answer', 'correct', 'rounds', 'trace_summary'
        """
        self.logger.reset()
        # Reset all agent histories
        for agent in self.agents:
            agent.reset()

        # Identify agent roles
        solver = self._get_agent("solver")
        critic = self._get_agent("critic")
        reviser = self._get_agent("reviser")
        verifier = self._get_agent("verifier")

        current_solution = ""
        final_answer: Optional[float] = None
        correct: Optional[bool] = None
        rounds_completed = 0

        for round_idx in range(self.max_rounds):
            rounds_completed = round_idx + 1

            # ── Step 1: Solve ──────────────────────────────────────────
            if solver:
                prompt = problem if round_idx == 0 else (
                    f"Problem: {problem}\n\n"
                    f"Previous solution had errors. Please re-solve:\n{current_solution}"
                )
                result = solver.call(prompt)
                current_solution = result["response"]
                self.logger.log_message(
                    solver.agent_id, solver.role, current_solution,
                    round_idx=round_idx, step_idx=0,
                    extra={"latency_ms": result["latency_ms"]},
                )

            # ── Step 2: Critique ───────────────────────────────────────
            verdict = "UNKNOWN"
            if critic:
                critique_result = critic.call(current_solution)
                critique = critique_result["response"]
                self.logger.log_message(
                    critic.agent_id, critic.role, critique,
                    round_idx=round_idx, step_idx=1,
                    extra={"latency_ms": critique_result["latency_ms"]},
                )
                if isinstance(critic, CriticAgent):
                    verdict = critic.extract_verdict(critique)
                elif "CORRECT" in critique.upper():
                    verdict = "CORRECT"
                elif "INCORRECT" in critique.upper():
                    verdict = "INCORRECT"

            # ── Early stop if CORRECT ──────────────────────────────────
            if verdict == "CORRECT":
                break

            # ── Step 3: Revise ─────────────────────────────────────────
            if reviser and verdict in ("INCORRECT", "UNCERTAIN", "UNKNOWN"):
                revise_prompt = (
                    f"Original solution:\n{current_solution}\n\n"
                    f"Critique:\n{critique if critic else 'Needs improvement.'}"
                )
                revise_result = reviser.call(revise_prompt)
                current_solution = revise_result["response"]
                self.logger.log_message(
                    reviser.agent_id, reviser.role, current_solution,
                    round_idx=round_idx, step_idx=2,
                    extra={"latency_ms": revise_result["latency_ms"]},
                )

        # ── Step 4: Verify final answer ────────────────────────────────
        if verifier:
            if isinstance(verifier, VerifierAgent):
                verification = verifier.verify(current_solution, ground_truth)
                final_answer = verification["extracted_answer"]
                correct = verification["correct"]
                self.logger.log_message(
                    verifier.agent_id, verifier.role, verification["response"],
                    round_idx=rounds_completed - 1, step_idx=3,
                )
            else:
                v_result = verifier.call(current_solution)
                self.logger.log_message(
                    verifier.agent_id, verifier.role, v_result["response"],
                    round_idx=rounds_completed - 1, step_idx=3,
                )

        return {
            "problem": problem,
            "final_solution": current_solution,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "correct": correct,
            "rounds": rounds_completed,
            "trace_summary": self.logger.get_summary(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_agent(self, role: str) -> Optional[BaseAgent]:
        """Return the first agent with matching role, or None."""
        for agent in self.agents:
            if agent.role == role:
                return agent
        return None


if __name__ == "__main__":
    from ..agents.solver_agent import SolverAgent
    from ..agents.critic_agent import CriticAgent as CA
    from ..agents.reviser_agent import ReviserAgent
    from ..agents.verifier_agent import VerifierAgent as VA

    pipeline = PipelineOrchestrator(
        agents=[SolverAgent(), CA(), ReviserAgent(), VA()],
        max_rounds=2,
    )
    result = pipeline.run("A store has 45 apples. It sells 18. How many remain?",
                          ground_truth=27.0)
    import json
    print(json.dumps(result, indent=2, default=str))
