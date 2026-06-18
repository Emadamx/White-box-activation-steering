"""
Metrics for steering evaluation.
"""
from __future__ import annotations
from typing import Dict


def compute_deception_rate(eval_results: Dict) -> float:
    """Fraction of episodes where the agent was unmonitored (deceptive mode active)."""
    return eval_results["deceptive_episodes"] / eval_results["n_episodes"]


def compute_performance_retention(current_perf: float, baseline_perf: float) -> float:
    """Fraction of baseline performance retained under the intervention."""
    if baseline_perf == 0:
        return 1.0
    return current_perf / baseline_perf
