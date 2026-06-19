"""
Experiment 03: Full Steering Evaluation.

Evaluates the steering intervention against baselines:
  - No intervention
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
    parser.add_argument("--direction-lp", default=None,
                        help="Path to LP direction .pt file. Auto-detected from exp02 if not set.")
    parser.add_argument("--direction-dom", default=None,
                        help="Path to DoM direction .pt file. Auto-detected from exp02 if not set.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index l*. Auto-read from exp01 results if not set.")
    parser.add_argument("--n-eval-episodes", type=int, default=200)
    parser.add_argument("--output-dir", default="results/exp03")
    return parser.parse_args()


def resolve_layer_and_paths(args):
    """Auto-detect l* and direction file paths from previous experiment results."""
    layer = args.layer

    # Read l* from exp01 results if not explicitly passed
    if layer is None:
        exp01_path = "results/exp01/results.json"
        if os.path.exists(exp01_path):
            with open(exp01_path) as f:
                exp01 = json.load(f)
            layer = exp01["best_layer"]
            print(f"Auto-detected l* = {layer} from Experiment 01 results.")
        else:
            layer = 0
            print(f"No Experiment 01 results found. Defaulting to layer {layer}.")

    # Build direction file paths using the actual layer index
    direction_lp  = args.direction_lp  or f"results/exp02/direction_lp_layer{layer}.pt"
    direction_dom = args.direction_dom or f"results/exp02/direction_dom_layer{layer}.pt"

    return layer, direction_lp, direction_dom


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Resolve layer and direction file paths automatically
    # ------------------------------------------------------------------ #
    layer_idx, direction_lp_path, direction_dom_path = resolve_layer_and_paths(args)

    # ------------------------------------------------------------------ #
    # Set up environment and agent
    # ------------------------------------------------------------------ #
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
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — using randomly initialised agent (for testing).")
    agent.eval()

    # ------------------------------------------------------------------ #
    # Baseline: no intervention
    # ------------------------------------------------------------------ #
    print("\nEvaluating baseline (no intervention)...")
    baseline_results = evaluate_agent(agent, env, n_episodes=args.n_eval_episodes)
    baseline_deception_rate = compute_deception_rate(baseline_results)
    baseline_performance    = baseline_results["mean_reward"]
    print(f"  Deception rate: {baseline_deception_rate:.2%}")
    print(f"  Mean reward:    {baseline_performance:.3f}")

    results_all = {
        "baseline": {
            "deception_rate": baseline_deception_rate,
            "performance":    baseline_performance,
        }
    }

    # ------------------------------------------------------------------ #
    # Activation steering sweep over alpha values
    # ------------------------------------------------------------------ #
    for method_name, direction_path in [("lp", direction_lp_path), ("dom", direction_dom_path)]:
        if not os.path.exists(direction_path):
            print(f"\nDirection file not found: {direction_path} — skipping {method_name}")
            continue

        direction = torch.load(direction_path, map_location="cpu")
        alpha_results = []

        print(f"\nEvaluating steering (method={method_name}, layer={layer_idx})...")
        for alpha in ALPHA_RANGE:
            intervention = SteeringIntervention(agent, direction, layer_idx, alpha=alpha)

            with intervention.active():
                eval_results = evaluate_agent(agent, env, n_episodes=args.n_eval_episodes)

            dr            = compute_deception_rate(eval_results)
            perf          = eval_results["mean_reward"]
            perf_retained = compute_performance_retention(perf, baseline_performance)

            alpha_results.append({
                "alpha":               alpha,
                "deception_rate":      dr,
                "performance":         perf,
                "performance_retained": perf_retained,
            })
            print(f"  α={alpha:.1f} | Deception: {dr:.2%} | Perf retained: {perf_retained:.2%}")

        results_all[f"steering_{method_name}"] = alpha_results

    # ------------------------------------------------------------------ #
    # Save results JSON
    # ------------------------------------------------------------------ #
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------ #
    # Plot safety-utility frontier (only if steering results exist)
    # ------------------------------------------------------------------ #
    steering_keys = [k for k in results_all if k.startswith("steering_")]
    if steering_keys:
        _plot_frontier(results_all, baseline_deception_rate, args.output_dir)
    else:
        print("\nNo steering results to plot (direction files were missing).")
        print("Run experiment 02 first, then re-run this experiment.")

    # ------------------------------------------------------------------ #
    # Print summary table always
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"{'Condition':<25} {'Deception Rate':>15} {'Perf Retained':>13}")
    print("-" * 55)
    print(f"{'No intervention':<25} {baseline_deception_rate:>14.2%} {'100.00%':>13}")
    for key in steering_keys:
        for entry in results_all[key]:
            label = f"{key.replace('steering_','').upper()} α={entry['alpha']:.1f}"
            print(f"  {label:<23} {entry['deception_rate']:>14.2%} {entry['performance_retained']:>13.2%}")


def _plot_frontier(results, baseline_dr, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter([baseline_dr], [1.0], marker="x", s=150, color="grey",
               zorder=5, label="No intervention")

    colors = {"steering_lp": "steelblue", "steering_dom": "darkorange"}
    for key, color in colors.items():
        if key not in results:
            continue
        data = results[key]
        drs   = [d["deception_rate"]      for d in data]
        perfs = [d["performance_retained"] for d in data]
        ax.plot(drs, perfs, marker="o", color=color, linewidth=2,
                label=key.replace("steering_", "Steering ").upper())
        for d in data:
            ax.annotate(
                f"α={d['alpha']:.1f}",
                xy=(d["deception_rate"], d["performance_retained"]),
                fontsize=7, ha="left", va="bottom", color=color,
            )

    ax.set_xlabel("Deception Rate ↓  (lower is safer)")
    ax.set_ylabel("Team Performance Retained ↑")
    ax.set_title("Safety–Utility Frontier: Activation Steering vs. Baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "safety_utility_frontier.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved safety-utility frontier plot to {plot_path}")


if __name__ == "__main__":
    main()
