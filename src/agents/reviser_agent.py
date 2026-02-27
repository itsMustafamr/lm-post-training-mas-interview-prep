"""
reviser_agent.py — Solution Reviser Agent
==========================================
The Reviser takes a (possibly incorrect) solution together with a
Critic's feedback and produces an improved, corrected solution.

It tracks revision count so downstream logic can detect infinite loops.
"""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent


class ReviserAgent(BaseAgent):
    """Agent that revises solutions based on critic feedback.

    Parameters
    ----------
    agent_id : str, optional
    model_name : str
    temperature : float
        Slightly higher than critic — need creative correction (default 0.7).
    max_new_tokens : int
    """

    SYSTEM_PROMPT = (
        "You are a math reviser. "
        "You will be given an original solution and a critique pointing out errors. "
        "Produce a corrected, complete solution that fixes all identified issues. "
        "Show all steps clearly and end with 'The answer is: [NUMBER]'."
    )

    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_name: str = "mock",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(
            agent_id=agent_id or "reviser_0",
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self.revision_count: int = 0

    @property
    def role(self) -> str:
        return "reviser"

    def generate(self, prompt: str) -> str:
        """Produce a revised solution given combined solution + critique in ``prompt``."""
        self.revision_count += 1
        if self.model_name == "mock":
            return self._mock_generate(prompt)
        return self._hf_generate(prompt)

    def reset(self) -> None:
        super().reset()
        self.revision_count = 0

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _mock_generate(self, prompt: str) -> str:
        return (
            f"[Revision #{self.revision_count}]\n"
            "Step 1: Re-read the problem carefully.\n"
            "Step 2: Identify the error noted in the critique.\n"
            "Step 3: Recompute with the correction applied.\n"
            "Step 4: Verify the result.\n"
            "The answer is: 27"
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
                    do_sample=True,
                )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}\n\nRevised solution:"
            output = self._model(full_prompt)[0]["generated_text"]
            return output[len(full_prompt) :].strip()
        except Exception as exc:
            return f"[ReviserAgent HF error: {exc}] " + self._mock_generate(prompt)


if __name__ == "__main__":
    reviser = ReviserAgent()
    prompt = (
        "Original solution: Step 1: 45 + 18 = 63. The answer is: 63\n"
        "Critique: VERDICT: INCORRECT — the problem asks for subtraction, not addition."
    )
    result = reviser.call(prompt)
    print("Revised:", result["response"])
    print("Revision count:", reviser.revision_count)
