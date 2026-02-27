"""
visualization.py — Plotting and Analysis Tools
===============================================
Provides matplotlib-based plots for:
  - Training loss/reward curves
  - Agent Shapley value / credit bar charts
  - Debate convergence over rounds
  - Credit attribution heatmaps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for Colab and servers)
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    losses: List[float],
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    smooth_window: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss (and optionally smoothed version).

    Parameters
    ----------
    losses : list[float] — per-step losses
    title : str
    smooth_window : int — rolling average window
    save_path : str, optional — if given, save to file

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = list(range(1, len(losses) + 1))

    ax.plot(steps, losses, alpha=0.3, color="steelblue", label="Raw loss")

    if smooth_window > 1 and len(losses) >= smooth_window:
        smoothed = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode="valid")
        smooth_steps = list(range(smooth_window, len(losses) + 1))
        ax.plot(smooth_steps, smoothed, color="steelblue", linewidth=2,
                label=f"Smoothed (w={smooth_window})")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_agent_contributions(
    shapley_values: Dict[str, float],
    title: str = "Agent Shapley Value Contributions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of Shapley values / credit for each agent.

    Parameters
    ----------
    shapley_values : dict  agent_id → shapley value
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    agents = list(shapley_values.keys())
    values = [shapley_values[a] for a in agents]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.2), 4))
    bars = ax.bar(agents, values, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(abs(v) for v in values),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Shapley Value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_debate_convergence(
    agreement_per_round: List[float],
    convergence_threshold: float = 0.67,
    title: str = "Multi-Agent Debate Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot showing agreement fraction across debate rounds.

    Parameters
    ----------
    agreement_per_round : list[float] — fraction of agents agreeing each round
    convergence_threshold : float — horizontal threshold line
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    rounds = list(range(1, len(agreement_per_round) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, agreement_per_round, "o-", color="darkorange",
            linewidth=2, markersize=8, label="Agreement fraction")
    ax.axhline(convergence_threshold, color="green", linestyle="--", linewidth=1.5,
               label=f"Convergence threshold ({convergence_threshold:.0%})")
    ax.fill_between(rounds, agreement_per_round, alpha=0.1, color="darkorange")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of agents in agreement")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(rounds)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_credit_heatmap(
    credit_matrix: List[List[float]],
    agent_labels: List[str],
    turn_labels: Optional[List[str]] = None,
    title: str = "Agent × Turn Credit Attribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of credit values for each (agent, turn) pair.

    Parameters
    ----------
    credit_matrix : list[list[float]]
        Shape: (n_agents, n_turns)
    agent_labels : list[str]
    turn_labels : list[str], optional
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    data = np.array(credit_matrix)
    n_agents, n_turns = data.shape

    if turn_labels is None:
        turn_labels = [f"Turn {t+1}" for t in range(n_turns)]

    fig, ax = plt.subplots(figsize=(max(8, n_turns * 1.2), max(4, n_agents * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Credit value")

    ax.set_xticks(range(n_turns))
    ax.set_xticklabels(turn_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agent_labels)
    ax.set_title(title, fontsize=14)

    # Annotate cells
    for i in range(n_agents):
        for j in range(n_turns):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import os

    # Training curve
    losses = [2.5 - 0.05 * i + 0.1 * np.random.randn() for i in range(100)]
    fig = plot_training_curves(losses, save_path="/tmp/training_curve.png")
    print("Saved training curve to /tmp/training_curve.png")
    plt.close(fig)

    # Shapley bar chart
    sv = {"solver": 0.45, "critic": 0.30, "reviser": 0.20, "verifier": 0.05}
    fig = plot_agent_contributions(sv, save_path="/tmp/shapley_bars.png")
    print("Saved Shapley bars to /tmp/shapley_bars.png")
    plt.close(fig)

    # Debate convergence
    agreement = [0.33, 0.55, 0.78, 0.89]
    fig = plot_debate_convergence(agreement, save_path="/tmp/debate_convergence.png")
    print("Saved debate convergence to /tmp/debate_convergence.png")
    plt.close(fig)

    # Credit heatmap
    matrix = [[0.8, 0.6, 0.9], [-0.3, 0.4, 0.5], [0.2, 0.7, 0.8]]
    fig = plot_credit_heatmap(matrix, ["solver", "critic", "reviser"],
                              save_path="/tmp/credit_heatmap.png")
    print("Saved credit heatmap to /tmp/credit_heatmap.png")
    plt.close(fig)
