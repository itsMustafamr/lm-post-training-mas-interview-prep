"""
solver_agent.py — Math Problem Solver Agent
============================================
The Solver is the first agent in a math-reasoning pipeline.
It receives a raw problem statement and produces a step-by-step solution.

In a real deployment this calls a fine-tuned LLM via the HuggingFace
``transformers`` API.  In demo / Colab mode it returns a templated
response so the pipeline can be exercised without a GPU.
"""

from __future__ import annotations

import re
from typing import Optional

from .base_agent import BaseAgent


class SolverAgent(BaseAgent):
    """Agent that solves math problems step by step.

    Parameters
    ----------
    agent_id : str, optional
        Unique identifier. Defaults to auto-generated.
    model_name : str
        HuggingFace model identifier or "mock" for demo mode.
    temperature : float
        Sampling temperature (higher = more creative solutions).
    max_new_tokens : int
        Maximum tokens to generate per response.
    """

    SYSTEM_PROMPT = (
        "You are an expert mathematician. "
        "Solve the given problem step by step, showing all reasoning clearly. "
        "End your response with: 'The answer is: [NUMBER]'"
    )

    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_name: str = "mock",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(
            agent_id=agent_id or "solver_0",
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self._model = None  # Lazy-loaded HuggingFace model

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def role(self) -> str:
        return "solver"

    def generate(self, prompt: str) -> str:
        """Generate a step-by-step solution for ``prompt``."""
        if self.model_name == "mock":
            return self._mock_generate(prompt)
        return self._hf_generate(prompt)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _mock_generate(self, prompt: str) -> str:
        """Template-based response for demos (no GPU required)."""
        # Try to extract numbers from the problem to make the demo realistic
        numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if len(numbers) >= 2:
            try:
                # Very simplistic: attempt addition of first two numbers
                answer = float(numbers[0]) + float(numbers[1])
                answer_str = str(int(answer)) if answer == int(answer) else str(answer)
            except ValueError:
                answer_str = "42"
        else:
            answer_str = "42"

        return (
            f"Step 1: Read the problem carefully.\n"
            f"Step 2: Identify the key values: {', '.join(numbers[:4]) if numbers else 'N/A'}.\n"
            f"Step 3: Apply the relevant operation.\n"
            f"Step 4: Compute the result.\n"
            f"The answer is: {answer_str}"
        )

    def _hf_generate(self, prompt: str) -> str:
        """Generate using a HuggingFace model (loads on first call)."""
        try:
            from transformers import pipeline  # type: ignore

            if self._model is None:
                self._model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nProblem: {prompt}\n\nSolution:"
            output = self._model(full_prompt)[0]["generated_text"]
            # Return only the newly generated part
            return output[len(full_prompt) :].strip()
        except Exception as exc:
            return f"[SolverAgent HF error: {exc}] " + self._mock_generate(prompt)


if __name__ == "__main__":
    solver = SolverAgent()
    problem = "A store has 45 apples. It sells 18 apples. How many are left?"
    result = solver.call(problem)
    print("Response:", result["response"])
    print("Latency:", result["latency_ms"], "ms")
    print(solver)
