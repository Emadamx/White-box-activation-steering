"""
Rollout utilities: collecting contrastive activation pairs and evaluating agents.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import numpy as np

from src.envs.deceptive_coin_game import DeceptiveCoinGame
from src.agents.transformer_agent import TransformerAgent
from src.steering.activation_hook import ActivationHook


def collect_contrastive_rollouts(
    agent: TransformerAgent,
    env: DeceptiveCoinGame,
    hook: ActivationHook,
    n_rollouts: int = 200,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Collect activation pairs from matched cooperative and deceptive rollouts.

    For each rollout, the same initial state is used with σ=1 (cooperative)
    and σ=0 (deceptive). Activations are collected at each hooked layer.

    Returns:
        (cooperative_activations, deceptive_activations)
        Each is {layer_idx: Tensor(n_rollouts, d_model)}.
    """
    coop_acts: Dict[int, List[torch.Tensor]] = {}
    dec_acts: Dict[int, List[torch.Tensor]] = {}

    for _ in range(n_rollouts):
        # Cooperative rollout (monitored)
        env.monitored = True
        obs_c = env.reset()
        obs_tensor_c = torch.tensor(obs_c[0], dtype=torch.float32).unsqueeze(0)
        with hook.capture():
            agent(obs_tensor_c)
        for layer_idx, act in hook.get_activations().items():
            coop_acts.setdefault(layer_idx, []).append(act)

        # Deceptive rollout (unmonitored) — same observation structure
        env.monitored = False
        obs_d = env.reset()
        obs_tensor_d = torch.tensor(obs_d[0], dtype=torch.float32).unsqueeze(0)
        # Zero out monitoring signal channel to simulate unmonitored
        obs_tensor_d[0, -1] = 0.0
        with hook.capture():
            agent(obs_tensor_d)
        for layer_idx, act in hook.get_activations().items():
            dec_acts.setdefault(layer_idx, []).append(act)

    # Stack into tensors
    coop_tensors = {k: torch.cat(v, dim=0) for k, v in coop_acts.items()}
    dec_tensors = {k: torch.cat(v, dim=0) for k, v in dec_acts.items()}

    return coop_tensors, dec_tensors


def evaluate_agent(
    agent: TransformerAgent,
    env: DeceptiveCoinGame,
    n_episodes: int = 100,
) -> Dict:
    """
    Evaluate the agent over n_episodes and return summary statistics.
    """
    all_rewards = []
    deceptive_episodes = 0

    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
            action = agent.act(obs_tensor)
            result = env.step({0: action, 1: 0})  # agent 1 stays (dummy)
            episode_reward += result.rewards[0]
            obs = result.observations
            done = result.done

        all_rewards.append(episode_reward)
        if not env.monitored:
            deceptive_episodes += 1

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "n_episodes": n_episodes,
        "deceptive_episodes": deceptive_episodes,
        "all_rewards": all_rewards,
    }
