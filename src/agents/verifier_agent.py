"""
verifier_agent.py — Final Answer Verifier Agent
================================================
The Verifier is the last agent in the pipeline. It:
  1. Extracts the final numerical answer from a solution text.
  2. Optionally checks it against a known ground-truth answer.
  3. Returns a structured result used for credit assignment.
"""

from __future__ import annotations

import re
from typing import Optional

from .base_agent import BaseAgent


class VerifierAgent(BaseAgent):
    """Agent that extracts and verifies final numerical answers.

    Parameters
    ----------
    agent_id : str, optional
    model_name : str
    temperature : float
        Very low (default 0.1) — extraction should be deterministic.
    """

    SYSTEM_PROMPT = (
        "You are a math answer verifier. "
        "Extract the final numerical answer from the given solution. "
        "Respond ONLY with: ANSWER: [NUMBER]"
    )

    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_name: str = "mock",
        temperature: float = 0.1,
        max_new_tokens: int = 32,
    ) -> None:
        super().__init__(
            agent_id=agent_id or "verifier_0",
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    @property
    def role(self) -> str:
        return "verifier"

    def generate(self, prompt: str) -> str:
        """Extract the answer from the solution text in ``prompt``."""
        if self.model_name == "mock":
            return self._mock_generate(prompt)
        return self._hf_generate(prompt)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def extract_answer(self, text: str) -> Optional[float]:
        """Parse 'ANSWER: NUMBER' from verifier output.

        Returns the float value, or None if not found.
        """
        match = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        # Fallback: look for 'The answer is: NUMBER'
        match2 = re.search(r"the answer is:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match2:
            return float(match2.group(1))
        return None

    def verify(self, solution_text: str, ground_truth: Optional[float] = None) -> dict:
        """Run extraction and optional comparison against ground truth.

        Returns
        -------
        dict
            Keys: 'extracted_answer', 'ground_truth', 'correct', 'response'
        """
        result = self.call(solution_text)
        extracted = self.extract_answer(result["response"])
        correct: Optional[bool] = None
        if ground_truth is not None and extracted is not None:
            correct = abs(extracted - ground_truth) < 1e-3
        return {
            "response": result["response"],
            "extracted_answer": extracted,
            "ground_truth": ground_truth,
            "correct": correct,
        }

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _mock_generate(self, prompt: str) -> str:
        # Try to extract the last number after 'answer is:'
        match = re.search(r"the answer is:\s*(-?\d+(?:\.\d+)?)", prompt, re.IGNORECASE)
        if match:
            return f"ANSWER: {match.group(1)}"
        # Fallback: last number in prompt
        numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
        if numbers:
            return f"ANSWER: {numbers[-1]}"
        return "ANSWER: unknown"

    def _hf_generate(self, prompt: str) -> str:
        try:
            from transformers import pipeline  # type: ignore

            if not hasattr(self, "_model") or self._model is None:
                self._model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nSolution:\n{prompt}\n\n"
            output = self._model(full_prompt)[0]["generated_text"]
            return output[len(full_prompt) :].strip()
        except Exception as exc:
            return f"[VerifierAgent HF error: {exc}] " + self._mock_generate(prompt)


if __name__ == "__main__":
    verifier = VerifierAgent()
    solution = "Step 1: 45 - 18 = 27.\nThe answer is: 27"
    result = verifier.verify(solution, ground_truth=27.0)
    print("Verification result:", result)
