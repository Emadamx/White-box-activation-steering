"""
Experiment 03: Full Steering Evaluation.

Evaluates the steering intervention against baselines:
  - No intervention
  - Behavioral monitor (output-level deception detection)
  - Activation steering (LP direction, DoM direction)

Produces the safety-utility frontier: deception rate vs. task performance retained.

Usage:
    python experiments/03_steering_evaluation.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from src.envs.deceptive_coin_game import DeceptiveCoinGame
from src.agents.transformer_agent import TransformerAgent
from src.steering.steering_intervention import SteeringIntervention
from src.utils.rollout import evaluate_agent
from src.utils.metrics import compute_deception_rate, compute_performance_retention


ALPHA_RANGE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--direction-lp", default="results/exp02/direction_lp_layer3.pt")
    parser.add_argument("--direction-dom", default="results/exp02/direction_dom_layer3.pt")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--n-eval-episodes", type=int, default=200)
    parser.add_argument("--output-dir", default="results/exp03")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = DeceptiveCoinGame(
        grid_size=cfg["env"]["grid_size"],
        max_steps=cfg["env"]["max_steps"],
        seed=cfg["seed"],
    )
    agent = TransformerAgent(
        obs_dim=env.observation_size,
        d_model=cfg["agent"]["d_model"],
        n_layers=cfg["agent"]["n_layers"],
        n_heads=cfg["agent"]["n_heads"],
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()

    # Baseline: no intervention
    print("Evaluating baseline (no intervention)...")
    baseline_results = evaluate_agent(agent, env, n_episodes=args.n_eval_episodes)
    baseline_deception_rate = compute_deception_rate(baseline_results)
    baseline_performance = baseline_results["mean_reward"]
    print(f"  Deception rate: {baseline_deception_rate:.2%}")
    print(f"  Mean reward:    {baseline_performance:.3f}")

    # Activation steering sweep over alpha values
    results_all = {"baseline": {"deception_rate": baseline_deception_rate,
                                 "performance": baseline_performance}}

    for method_name, direction_path in [("lp", args.direction_lp), ("dom", args.direction_dom)]:
        if not os.path.exists(direction_path):
            print(f"Direction file not found: {direction_path} — skipping {method_name}")
            continue

        direction = torch.load(direction_path)
        alpha_results = []

        for alpha in ALPHA_RANGE:
            intervention = SteeringIntervention(agent, direction, args.layer, alpha=alpha)
            print(f"Evaluating steering ({method_name}, α={alpha:.1f})...")

            with intervention.active():
                eval_results = evaluate_agent(agent, env, n_episodes=args.n_eval_episodes)

            dr = compute_deception_rate(eval_results)
            perf = eval_results["mean_reward"]
            perf_retained = compute_performance_retention(perf, baseline_performance)

            alpha_results.append({
                "alpha": alpha,
                "deception_rate": dr,
                "performance": perf,
                "performance_retained": perf_retained,
            })
            print(f"  α={alpha:.1f} | Deception: {dr:.2%} | Perf retained: {perf_retained:.2%}")

        results_all[f"steering_{method_name}"] = alpha_results

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_all, f, indent=2)

    # Plot safety-utility frontier
    _plot_frontier(results_all, baseline_deception_rate, baseline_performance, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


def _plot_frontier(results, baseline_dr, baseline_perf, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter([baseline_dr], [1.0], marker="x", s=150, color="grey",
               zorder=5, label="No intervention")

    colors = {"steering_lp": "steelblue", "steering_dom": "darkorange"}
    for key, color in colors.items():
        if key not in results:
            continue
        data = results[key]
        drs = [d["deception_rate"] for d in data]
        perfs = [d["performance_retained"] for d in data]
        ax.plot(drs, perfs, marker="o", color=color, linewidth=2,
                label=key.replace("steering_", "Steering ").upper())
        # Label alpha values
        for d in data:
            ax.annotate(
                f"α={d['alpha']:.1f}",
                xy=(d["deception_rate"], d["performance_retained"]),
                fontsize=7, ha="left", va="bottom", color=color,
            )

    ax.set_xlabel("Deception Rate ↓")
    ax.set_ylabel("Team Performance Retained ↑")
    ax.set_title("Safety–Utility Frontier: Activation Steering vs. Baselines")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "safety_utility_frontier.png"), dpi=150)
    print("Saved safety-utility frontier plot.")


if __name__ == "__main__":
    main()
