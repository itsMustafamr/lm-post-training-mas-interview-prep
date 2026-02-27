"""
critic_agent.py — Solution Critic / Reviewer Agent
====================================================
The Critic reviews a proposed solution and outputs a structured verdict.

Verdict format (used by downstream agents and credit assignment):
  VERDICT: CORRECT   — solution is logically sound
  VERDICT: INCORRECT — solution contains an error (with explanation)
  VERDICT: UNCERTAIN — not enough information to judge
"""

from __future__ import annotations

import re
from typing import Optional

from .base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """Agent that reviews math solutions for correctness.

    Parameters
    ----------
    agent_id : str, optional
        Unique identifier.
    model_name : str
        HuggingFace model or "mock" for demo mode.
    temperature : float
        Lower temperature → more deterministic critiques (default 0.3).
    """

    SYSTEM_PROMPT = (
        "You are a rigorous math critic. "
        "Review the provided solution for logical errors, arithmetic mistakes, "
        "or missing steps. "
        "Your response must start with one of:\n"
        "  VERDICT: CORRECT\n"
        "  VERDICT: INCORRECT\n"
        "  VERDICT: UNCERTAIN\n"
        "Then explain your reasoning."
    )

    VERDICTS = ("CORRECT", "INCORRECT", "UNCERTAIN")

    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_name: str = "mock",
        temperature: float = 0.3,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            agent_id=agent_id or "critic_0",
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    @property
    def role(self) -> str:
        return "critic"

    def generate(self, prompt: str) -> str:
        """Review the solution in ``prompt`` and return a verdict."""
        if self.model_name == "mock":
            return self._mock_generate(prompt)
        return self._hf_generate(prompt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def extract_verdict(self, response: str) -> str:
        """Parse the VERDICT line from a critic response.

        Returns one of 'CORRECT', 'INCORRECT', 'UNCERTAIN', or 'UNKNOWN'.
        """
        for v in self.VERDICTS:
            if re.search(rf"\bVERDICT:\s*{v}\b", response, re.IGNORECASE):
                return v
        return "UNKNOWN"

    def _mock_generate(self, prompt: str) -> str:
        """Simple heuristic critique for demos."""
        # Heuristic: if the solution mentions 'The answer is:' it looks complete
        if "the answer is:" in prompt.lower():
            return (
                "VERDICT: CORRECT\n"
                "The solution follows a clear step-by-step approach and arrives "
                "at a definitive numerical answer. No logical errors detected."
            )
        return (
            "VERDICT: UNCERTAIN\n"
            "The solution does not end with a clear numerical answer. "
            "Please revise to include 'The answer is: [NUMBER]'."
        )

    def _hf_generate(self, prompt: str) -> str:
        try:
            from transformers import pipeline  # type: ignore

            if not hasattr(self, "_model") or self._model is None:
                self._model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
            full_prompt = (
                f"{self.SYSTEM_PROMPT}\n\nSolution to review:\n{prompt}\n\nCritique:"
            )
            output = self._model(full_prompt)[0]["generated_text"]
            return output[len(full_prompt) :].strip()
        except Exception as exc:
            return f"[CriticAgent HF error: {exc}] " + self._mock_generate(prompt)


if __name__ == "__main__":
    critic = CriticAgent()
    solution = "Step 1: 45 - 18 = 27.\nThe answer is: 27"
    result = critic.call(solution)
    print("Critique:", result["response"])
    print("Verdict:", critic.extract_verdict(result["response"]))
