"""
Evaluation sub-package
=======================
"""

from .benchmarks import GSM8K_SAMPLE, MATH_SAMPLE, evaluate_on_benchmark, extract_numerical_answer
from .metrics import MetricsTracker, compute_accuracy, compute_convergence_rate
from .visualization import (
    plot_training_curves,
    plot_agent_contributions,
    plot_debate_convergence,
    plot_credit_heatmap,
)

__all__ = [
    "GSM8K_SAMPLE", "MATH_SAMPLE", "evaluate_on_benchmark", "extract_numerical_answer",
    "MetricsTracker", "compute_accuracy", "compute_convergence_rate",
    "plot_training_curves", "plot_agent_contributions",
    "plot_debate_convergence", "plot_credit_heatmap",
]
