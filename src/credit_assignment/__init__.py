"""
Credit assignment sub-package
==============================
Tools for attributing system-level outcomes to individual agents.
"""

from .shapley import ShapleyCalculator, exact_shapley_values, approximate_shapley_values
from .error_localization import ErrorLocalizer, locate_first_error
from .process_reward import StepRewardModel, assign_step_rewards
from .at_grpo import ATGRPOConfig, ATGRPOTrainer

__all__ = [
    "ShapleyCalculator", "exact_shapley_values", "approximate_shapley_values",
    "ErrorLocalizer", "locate_first_error",
    "StepRewardModel", "assign_step_rewards",
    "ATGRPOConfig", "ATGRPOTrainer",
]
