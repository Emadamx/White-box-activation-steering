"""
Experiment 02: Steering Direction Learning.

Compares three direction-learning methods (LP, DoM, PCA) at the best layer l*
identified in Experiment 01.

Usage:
    python experiments/02_direction_learning.py --config configs/default.yaml \
        --method lp --layer 3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np

from src.envs.deceptive_coin_game import DeceptiveCoinGame
from src.agents.transformer_agent import TransformerAgent
from src.steering.activation_hook import ActivationHook
from src.steering.direction_finder import DirectionFinder
from src.utils.rollout import collect_contrastive_rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["lp", "dom", "pca"], default="lp")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index l*. If not set, reads from exp01 results.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-rollouts", type=int, default=300)
    parser.add_argument("--output-dir", default="results/exp02")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve best layer
    layer_idx = args.layer
    if layer_idx is None:
        exp01_path = "results/exp01/results.json"
        if os.path.exists(exp01_path):
            with open(exp01_path) as f:
                exp01 = json.load(f)
            layer_idx = exp01["best_layer"]
            print(f"Using l* = {layer_idx} from Experiment 01 results.")
        else:
            layer_idx = cfg["agent"]["n_layers"] // 2
            print(f"No Experiment 01 results found. Defaulting to middle layer: {layer_idx}")

    # Set up env and agent
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

    # Collect rollouts
    hook = ActivationHook(agent, layer_indices=[layer_idx])
    print(f"Collecting {args.n_rollouts} contrastive rollout pairs at layer {layer_idx}...")
    coop_acts, dec_acts = collect_contrastive_rollouts(
        agent=agent, env=env, hook=hook, n_rollouts=args.n_rollouts
    )

    H_c = coop_acts[layer_idx]
    H_d = dec_acts[layer_idx]

    # Learn direction
    print(f"Learning direction with method='{args.method}'...")
    finder = DirectionFinder(method=args.method, normalize=True)
    finder.fit(H_c, H_d)

    print(f"Direction shape: {finder.direction.shape}")
    print(f"Direction norm: {np.linalg.norm(finder.direction):.4f} (should be 1.0)")
    if args.method == "lp":
        print(f"Linear probe train accuracy: {finder.probe_accuracy_:.3f}")

    # Save
    direction_path = os.path.join(args.output_dir, f"direction_{args.method}_layer{layer_idx}.pt")
    torch.save(finder.to_tensor(), direction_path)
    print(f"Saved direction to {direction_path}")

    results = {
        "method": args.method,
        "layer": layer_idx,
        "probe_accuracy": finder.probe_accuracy_,
        "direction_path": direction_path,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
