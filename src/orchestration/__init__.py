"""
Orchestration sub-package
=========================
Pipeline patterns for coordinating multiple agents.
"""

from .logger import TraceLogger
from .pipeline import PipelineOrchestrator
from .debate import DebateOrchestrator
from .mixture_of_agents import MixtureOfAgents

__all__ = [
    "TraceLogger",
    "PipelineOrchestrator",
    "DebateOrchestrator",
    "MixtureOfAgents",
]
