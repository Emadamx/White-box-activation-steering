"""
Experiment 01: Layer Selection via Linear Deception Probes.

This experiment:
  1. Loads a trained deceptive agent.
  2. Collects cooperative (σ=1) and deceptive (σ=0) rollouts.
  3. Trains a linear probe at each transformer layer.
  4. Plots probe accuracy vs. layer depth.
  5. Identifies l* — the best layer for direction learning.

Usage:
    python experiments/01_probe_layer_selection.py --config configs/default.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import matplotlib.pyplot as plt

from src.envs.deceptive_coin_game import DeceptiveCoinGame
from src.agents.transformer_agent import TransformerAgent
from src.steering.activation_hook import ActivationHook
from src.probing.linear_probe import LinearDeceptionProbe
from src.utils.rollout import collect_contrastive_rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to trained agent checkpoint.")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--output-dir", default="results/exp01")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Set up environment and agent
    # ------------------------------------------------------------------ #
    env = DeceptiveCoinGame(
        grid_size=cfg["env"]["grid_size"],
        max_steps=cfg["env"]["max_steps"],
        seed=cfg["seed"],
    )

    agent = TransformerAgent(
        obs_dim=env.observation_size,
        n_actions=5,
        d_model=cfg["agent"]["d_model"],
        n_layers=cfg["agent"]["n_layers"],
        n_heads=cfg["agent"]["n_heads"],
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint found — using randomly initialised agent (for testing).")

    agent.eval()

    # ------------------------------------------------------------------ #
    # 2. Collect contrastive rollouts
    # ------------------------------------------------------------------ #
    print(f"Collecting {args.n_rollouts} contrastive rollout pairs...")
    hook = ActivationHook(agent, layer_indices=list(range(cfg["agent"]["n_layers"])))

    cooperative_acts, deceptive_acts = collect_contrastive_rollouts(
        agent=agent,
        env=env,
        hook=hook,
        n_rollouts=args.n_rollouts,
    )

    # ------------------------------------------------------------------ #
    # 3. Train layer-wise probes
    # ------------------------------------------------------------------ #
    print("Training linear deception probes at each layer...")
    probe = LinearDeceptionProbe(cv_folds=5, random_state=cfg["seed"])
    probe.fit_all_layers(cooperative_acts, deceptive_acts)

    print(probe.summary())
    print(f"\nBest layer: l* = {probe.best_layer_} "
          f"(accuracy={probe.layer_accuracies_[probe.best_layer_]:.3f})")

    # ------------------------------------------------------------------ #
    # 4. Plot and save
    # ------------------------------------------------------------------ #
    layers, accs = probe.get_accuracy_curve()
    plt.figure(figsize=(8, 4))
    plt.plot(layers, accs, marker="o", linewidth=2)
    plt.axvline(probe.best_layer_, color="red", linestyle="--", label=f"l* = {probe.best_layer_}")
    plt.axhline(0.5, color="grey", linestyle=":", label="Chance")
    plt.xlabel("Layer Index")
    plt.ylabel("Probe Accuracy (5-fold CV)")
    plt.title("Linear Deception Probe Accuracy vs. Transformer Layer")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "probe_accuracy_curve.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # Save results
    results = {
        "best_layer": probe.best_layer_,
        "layer_accuracies": probe.layer_accuracies_,
    }
    import json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
