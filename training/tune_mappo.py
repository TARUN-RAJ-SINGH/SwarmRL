"""
SwarmRL -- Day 18: Automated MAPPO Hyperparameter Tuning & Grid Search

Systematic search and multi-objective Pareto analysis across key MAPPO hyperparameters:
  - Critic-to-actor learning rate ratio (eta_critic / eta_actor)
  - Entropy regularization coefficient (beta_ent)
  - PPO clipping parameter (epsilon)
  - GAE discount lambda

Evaluates and ranks configurations based on:
  1. Area Coverage % (Exploration efficacy)
  2. Total Collisions (Safety & cooperative clearance)
  3. Critic Explained Variance (Value baseline fidelity)
  4. Cumulative Return
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from env.spaces import OBS_DIM
from env.gnn_comm import ENHANCED_OBS_DIM
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer


DEFAULT_PARAM_GRID = {
    "lr_actor": [1e-4, 3e-4],
    "lr_ratio": [1.0, 3.33, 5.0],      # lr_critic = lr_actor * lr_ratio
    "entropy_coef": [0.005, 0.01, 0.02],
    "clip_param": [0.1, 0.2],
    "gae_lambda": [0.90, 0.95],
}


def sample_hyperparameters(grid: Dict[str, list], num_samples: int = 4, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates reproducible hyperparameter configurations sampled from parameter grid."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(num_samples):
        lr_actor = float(rng.choice(grid["lr_actor"]))
        ratio = float(rng.choice(grid["lr_ratio"]))
        lr_critic = lr_actor * ratio
        entropy_coef = float(rng.choice(grid["entropy_coef"]))
        clip_param = float(rng.choice(grid["clip_param"]))
        gae_lambda = float(rng.choice(grid["gae_lambda"]))

        configs.append({
            "trial_id": i + 1,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "lr_ratio": ratio,
            "entropy_coef": entropy_coef,
            "clip_param": clip_param,
            "gae_lambda": gae_lambda,
        })
    return configs


def run_trial(
    params: Dict[str, Any],
    num_drones: int = 10,
    iterations: int = 5,
    steps_per_iter: int = 50,
    use_comm: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes a single tuning trial for a fixed iteration budget.

    Returns:
        Trial metrics dictionary with coverage, reward, collisions, explained variance, and duration.
    """
    obs_dim = ENHANCED_OBS_DIM if use_comm else OBS_DIM
    if use_comm:
        env = CommSwarmEnv(num_drones=num_drones, loss_prob=0.0, formation_bonus=True)
    else:
        env = SwarmEnv(num_drones=num_drones)

    config = MAPPOConfig(
        lr_actor=params["lr_actor"],
        lr_critic=params["lr_critic"],
        entropy_coef=params["entropy_coef"],
        clip_param=params["clip_param"],
        gae_lambda=params["gae_lambda"],
        ppo_epochs=3,
        actor_mini_batch_size=32,
        critic_mini_batch_size=16,
        use_popart=True,
    )

    trainer = MAPPOTrainer(
        config=config,
        obs_dim=obs_dim,
        num_agents=num_drones,
        rollout_steps=steps_per_iter,
        device=device,
    )

    t0 = time.time()
    rewards = []
    coverages = []
    collisions = []
    explained_vars = []

    for it in range(iterations):
        res = trainer.train_iteration(env, num_steps=steps_per_iter)
        rewards.append(res["mean_reward"])
        coverages.append(res["final_coverage"])
        collisions.append(res["collisions"])
        explained_vars.append(res["explained_variance"])

    duration = time.time() - t0

    return {
        **params,
        "final_coverage": float(coverages[-1]),
        "mean_coverage": float(np.mean(coverages)),
        "final_reward": float(rewards[-1]),
        "mean_reward": float(np.mean(rewards)),
        "total_collisions": int(np.sum(collisions)),
        "mean_explained_variance": float(np.mean(explained_vars)),
        "duration_seconds": float(duration),
    }


def compute_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies non-dominated trials considering:
      - Higher coverage is better
      - Lower collisions is better
      - Higher mean reward is better
    """
    pareto_optimal = []
    for i, candidate in enumerate(results):
        dominated = False
        c_cov = candidate["final_coverage"]
        c_col = candidate["total_collisions"]
        c_rew = candidate["mean_reward"]

        for j, other in enumerate(results):
            if i == j:
                continue
            o_cov = other["final_coverage"]
            o_col = other["total_collisions"]
            o_rew = other["mean_reward"]

            # other dominates candidate if it is >= in all objectives and strictly > in at least one
            if (o_cov >= c_cov and o_col <= c_col and o_rew >= c_rew) and \
               (o_cov > c_cov or o_col < c_col or o_rew > c_rew):
                dominated = True
                break

        if not dominated:
            candidate_copy = dict(candidate)
            candidate_copy["is_pareto"] = True
            pareto_optimal.append(candidate_copy)

    return pareto_optimal


def main():
    parser = argparse.ArgumentParser(description="SwarmRL MAPPO Hyperparameter Tuning")
    parser.add_argument("--trials", type=int, default=4, help="Number of trials to sample (default: 4)")
    parser.add_argument("--iterations", type=int, default=5, help="Training iterations per trial (default: 5)")
    parser.add_argument("--steps", type=int, default=50, help="Steps per iteration (default: 50)")
    parser.add_argument("--drones", type=int, default=10, help="Fleet size (default: 10)")
    parser.add_argument("--no-comm", action="store_true", help="Disable GNN comm environment")
    parser.add_argument("--output", type=str, default="training/hyperparameter_tuning_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("           SwarmRL — Automated MAPPO Hyperparameter Search & Tuning            ")
    print("=" * 80)
    print(f" Sampling {args.trials} configurations | {args.iterations} iterations/trial | {args.drones} drones")
    print("-" * 80)

    configs = sample_hyperparameters(DEFAULT_PARAM_GRID, num_samples=args.trials, seed=42)
    results = []

    for cfg in configs:
        t_id = cfg["trial_id"]
        print(f"Running Trial {t_id}/{args.trials}: lr_act={cfg['lr_actor']}, ratio={cfg['lr_ratio']}, "
              f"ent={cfg['entropy_coef']}, clip={cfg['clip_param']} ...", end=" ", flush=True)

        res = run_trial(
            params=cfg,
            num_drones=args.drones,
            iterations=args.iterations,
            steps_per_iter=args.steps,
            use_comm=not args.no_comm,
        )
        results.append(res)
        print(f"Done in {res['duration_seconds']:.2f}s | Cov: {res['final_coverage']*100:.1f}% | Cols: {res['total_collisions']}")

    pareto = compute_pareto_frontier(results)
    best_trial = max(results, key=lambda x: x["final_coverage"])

    summary = {
        "num_trials": len(results),
        "best_trial_by_coverage": best_trial,
        "pareto_frontier_count": len(pareto),
        "pareto_frontier": pareto,
        "all_trials": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Best Configuration: Trial #{best_trial['trial_id']}")
    print(f"  Coverage: {best_trial['final_coverage']*100:.1f}% | Mean Reward: {best_trial['mean_reward']:.3f} | Collisions: {best_trial['total_collisions']}")
    print(f"  Parameters: lr_actor={best_trial['lr_actor']}, lr_critic={best_trial['lr_critic']}, "
          f"entropy_coef={best_trial['entropy_coef']}, clip_param={best_trial['clip_param']}")
    print(f"Results written to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
