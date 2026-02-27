"""
logger.py — Comprehensive Trace Logging System
===============================================
Records every message exchanged in a multi-agent pipeline with:
  - Timestamps (ISO-8601)
  - Agent ID and role
  - Message content
  - Estimated token counts
  - Round and step indices

Output can be exported to JSON for later analysis or credit assignment.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


class TraceLogger:
    """Logs the complete trace of a multi-agent interaction.

    Usage
    -----
    >>> logger = TraceLogger(session_id="run_001")
    >>> logger.log_message(agent_id="solver_0", role="solver",
    ...                    content="Step 1: ...", round_idx=0, step_idx=0)
    >>> logger.save_to_json("trace.json")
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or f"session_{int(time.time())}"
        self.messages: List[Dict] = []
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_message(
        self,
        agent_id: str,
        role: str,
        content: str,
        round_idx: int = 0,
        step_idx: int = 0,
        extra: Optional[Dict] = None,
    ) -> None:
        """Record a single message in the trace.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier.
        role : str
            Agent role (solver, critic, etc.).
        content : str
            The message text.
        round_idx : int
            Which debate/pipeline round this belongs to.
        step_idx : int
            Which step within the round.
        extra : dict, optional
            Any additional metadata (latency, model name, etc.).
        """
        entry: Dict = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "round": round_idx,
            "step": step_idx,
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "token_estimate": _estimate_tokens(content),
        }
        if extra:
            entry.update(extra)
        self.messages.append(entry)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_trace(self) -> List[Dict]:
        """Return the full message trace."""
        return list(self.messages)

    def get_messages_by_agent(self, agent_id: str) -> List[Dict]:
        """Filter messages by agent ID."""
        return [m for m in self.messages if m["agent_id"] == agent_id]

    def get_messages_by_role(self, role: str) -> List[Dict]:
        """Filter messages by role."""
        return [m for m in self.messages if m["role"] == role]

    def get_summary(self) -> Dict:
        """Return summary statistics for the trace."""
        if not self.messages:
            return {"total_messages": 0}
        roles = {}
        for m in self.messages:
            roles[m["role"]] = roles.get(m["role"], 0) + 1
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "total_tokens_estimate": sum(m["token_estimate"] for m in self.messages),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "messages_per_role": roles,
            "num_rounds": max(m["round"] for m in self.messages) + 1,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_json(self, path: str) -> None:
        """Write the full trace to a JSON file."""
        output = {
            "summary": self.get_summary(),
            "messages": self.messages,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def reset(self) -> None:
        """Clear all logged messages."""
        self.messages = []
        self._start_time = time.time()


if __name__ == "__main__":
    logger = TraceLogger(session_id="demo_001")
    logger.log_message("solver_0", "solver", "Step 1: 45 - 18 = 27. The answer is: 27",
                       round_idx=0, step_idx=0)
    logger.log_message("critic_0", "critic", "VERDICT: CORRECT — arithmetic is sound.",
                       round_idx=0, step_idx=1)
    print(json.dumps(logger.get_summary(), indent=2))
    logger.save_to_json("/tmp/demo_trace.json")
    print("Saved to /tmp/demo_trace.json")
