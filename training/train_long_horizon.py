"""
SwarmRL -- Day 19: Long-Horizon Curriculum-Guided MAPPO Training

Trains a robust multi-agent drone swarm policy across all curriculum stages:
  - Stage 1: Base Camp Dispersal & Obstacle Avoidance
  - Stage 2: Priority Zone A Sector Exploration & Formation Flight
  - Stage 3: Full 3D Disaster Zone Coverage (Zone A + Zone B)

Automatically tracks performance metrics, triggers stage promotions,
and exports the top-performing production model to checkpoints/day19_mappo_best.pt.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.comm_swarm_env import CommSwarmEnv
from env.gnn_comm import ENHANCED_OBS_DIM
from training.curriculum import CurriculumManager
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer


def train_long_horizon(
    num_drones: int = 10,
    total_iterations: int = 15,
    steps_per_iter: int = 100,
    checkpoint_output: str = "checkpoints/day19_mappo_best.pt",
    seed: int = 42,
    device: str = "cpu",
):
    """
    Executes curriculum-guided long-horizon MAPPO training.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print("           SwarmRL — Long-Horizon Curriculum MAPPO Training                    ")
    print("=" * 80)
    print(f" Fleet Size          : {num_drones} drones")
    print(f" Budget              : {total_iterations} iterations x {steps_per_iter} steps")
    print(f" Environment         : CommSwarmEnv (42-dim GNN spatial attention)")
    print(f" Target Checkpoint   : {checkpoint_output}")
    print("=" * 80)

    # Initialize environment and curriculum manager
    env = CommSwarmEnv(num_drones=num_drones, loss_prob=0.0, formation_bonus=True)
    curriculum = CurriculumManager()
    curriculum.apply_to_env(env)

    config = MAPPOConfig(
        lr_actor=3e-4,
        lr_critic=1e-3,
        ppo_epochs=5,
        actor_mini_batch_size=64,
        critic_mini_batch_size=16,
        entropy_coef=0.01,
        critic_type="attention",
        use_popart=True,
    )

    trainer = MAPPOTrainer(
        config=config,
        obs_dim=ENHANCED_OBS_DIM,
        num_agents=num_drones,
        rollout_steps=steps_per_iter,
        device=device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_output)), exist_ok=True)

    best_coverage = -1.0
    start_time = time.time()

    print(f"\n[Curriculum Initialized] Stage 1: {curriculum.current_stage.name}")
    print("-" * 88)
    print(f"{'Iter':>5} | {'Stage':>7} | {'Reward':>8} | {'Cover %':>7} | {'Zone A':>6} | {'Cols':>4} | {'Pol Loss':>8} | {'Val Loss':>8} | {'Time':>6}")
    print("-" * 88)

    for it in range(1, total_iterations + 1):
        t0 = time.time()
        metrics = trainer.train_iteration(env, num_steps=steps_per_iter)
        dt = time.time() - t0

        cov = metrics["final_coverage"]
        za_cov = metrics["zone_a_coverage"]
        stage_name = f"S{curriculum.current_stage.stage_id}"

        print(
            f"{it:5d} | "
            f"{stage_name:>7} | "
            f"{metrics['mean_reward']:8.3f} | "
            f"{cov * 100:6.1f}% | "
            f"{za_cov * 100:5.1f}% | "
            f"{metrics['collisions']:4d} | "
            f"{metrics['policy_loss']:8.4f} | "
            f"{metrics['value_loss']:8.4f} | "
            f"{dt:5.2f}s"
        )

        # Check curriculum progression
        promoted, reason = curriculum.evaluate_progress(metrics)
        if promoted:
            print(f"\n>>> [CURRICULUM PROMOTION] {reason}")
            curriculum.apply_to_env(env)
            print("-" * 88)

        # Save best model
        if cov > best_coverage:
            best_coverage = cov
            trainer.save_checkpoint(checkpoint_output)

    total_time = time.time() - start_time
    print("-" * 88)
    print(f"Long-horizon training finished in {total_time:.2f}s!")
    print(f"Peak Coverage Achieved: {best_coverage * 100:.1f}%")
    print(f"Production model saved to: {checkpoint_output}")
    print("=" * 80)
    return best_coverage


def main():
    parser = argparse.ArgumentParser(description="SwarmRL Long-Horizon Training")
    parser.add_argument("--drones", type=int, default=10, help="Fleet size (default: 10)")
    parser.add_argument("--iterations", type=int, default=15, help="Total training iterations (default: 15)")
    parser.add_argument("--steps", type=int, default=80, help="Steps per rollout iteration (default: 80)")
    parser.add_argument("--output", type=str, default="checkpoints/day19_mappo_best.pt", help="Best checkpoint output path")
    args = parser.parse_args()

    train_long_horizon(
        num_drones=args.drones,
        total_iterations=args.iterations,
        steps_per_iter=args.steps,
        checkpoint_output=args.output,
    )


if __name__ == "__main__":
    main()
