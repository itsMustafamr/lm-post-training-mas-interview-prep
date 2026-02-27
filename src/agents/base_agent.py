"""
base_agent.py — Abstract Base Class for All Agents
===================================================
Theory
------
In a multi-agent system (MAS) each agent is an autonomous entity that:
  1. Perceives its environment (the conversation / problem context)
  2. Reasons about that context
  3. Produces an action (a text response)

This module defines the contract every agent must fulfill, allowing
orchestrators to treat different agent types uniformly.

Implementation Notes
--------------------
- Uses Python ABC so that forgetting to implement `generate()` raises a clear error.
- Message history is stored as a list of OpenAI-style dicts:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
- A "mock backend" is used when no real LLM is available, enabling demos on
  any machine without GPU or API keys.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseAgent(ABC):
    """Abstract base class for all agents in the MAS pipeline.

    Subclasses must implement:
      - ``role`` property  — string identifier, e.g. "solver"
      - ``generate(prompt)`` — produce a text response given a prompt

    Attributes
    ----------
    agent_id : str
        Unique identifier for this agent instance.
    model_name : str
        Name of the underlying language model.
    history : list[dict]
        Conversation history in OpenAI message format.
    _call_count : int
        Number of times ``generate()`` has been called.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_name: str = "mock",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> None:
        self.agent_id: str = agent_id or f"{self.__class__.__name__}_{id(self)}"
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_new_tokens: int = max_new_tokens
        self.history: List[Dict[str, str]] = []
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must provide these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def role(self) -> str:
        """String role identifier, e.g. 'solver', 'critic', 'verifier'."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response given the current prompt.

        Parameters
        ----------
        prompt : str
            The user-facing input (problem statement, critique, etc.)

        Returns
        -------
        str
            The agent's response text.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def add_to_history(self, role: str, content: str) -> None:
        """Append a message to this agent's conversation history."""
        self.history.append({"role": role, "content": content})

    def reset(self) -> None:
        """Clear history and reset call counter (start a new problem)."""
        self.history = []
        self._call_count = 0

    def call(self, prompt: str) -> Dict[str, object]:
        """Wrapper around ``generate()`` that tracks timing and history.

        Returns a dict with keys: response, latency_ms, call_index.
        """
        self.add_to_history("user", prompt)
        start = time.time()
        response = self.generate(prompt)
        latency_ms = (time.time() - start) * 1000
        self.add_to_history("assistant", response)
        self._call_count += 1
        return {
            "response": response,
            "latency_ms": round(latency_ms, 2),
            "call_index": self._call_count,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id!r}, "
            f"role={self.role!r}, "
            f"model={self.model_name!r}, "
            f"calls={self._call_count})"
        )


if __name__ == "__main__":
    # Quick smoke-test: concrete minimal subclass
    class EchoAgent(BaseAgent):
        @property
        def role(self) -> str:
            return "echo"

        def generate(self, prompt: str) -> str:
            return f"Echo: {prompt}"

    agent = EchoAgent(agent_id="echo_0")
    result = agent.call("Hello, world!")
    print(result)
    print(agent)
