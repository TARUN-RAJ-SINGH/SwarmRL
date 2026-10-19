"""
SwarmRL -- Day 18: Fleet Scalability Training Benchmark

Benchmarks MAPPO CTDE across variable fleet sizes (10, 25, 50 drones):
  - Step throughput & Physics FPS
  - Centralized critic inference latency
  - Swarm coverage rate progression
  - Inter-drone collision density
  - Exports structured performance logs to training/scaling_results.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.comm_swarm_env import CommSwarmEnv
from env.gnn_comm import ENHANCED_OBS_DIM
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer


def benchmark_fleet_scale(
    fleet_sizes: List[int] = [10, 25, 50],
    iterations: int = 3,
    steps_per_iter: int = 40,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes comparative MAPPO training iterations across specified fleet sizes.

    Returns:
        Structured dictionary containing metrics per fleet size and scalability summaries.
    """
    results = {}

    for n in fleet_sizes:
        print(f"\nBenchmarking Fleet Size N = {n} drones...")
        env = CommSwarmEnv(num_drones=n, loss_prob=0.0, formation_bonus=True)

        config = MAPPOConfig(
            lr_actor=3e-4,
            lr_critic=1e-3,
            ppo_epochs=2,
            actor_mini_batch_size=min(64, n * 10),
            critic_mini_batch_size=16,
            use_popart=True,
        )

        trainer = MAPPOTrainer(
            config=config,
            obs_dim=ENHANCED_OBS_DIM,
            num_agents=n,
            rollout_steps=steps_per_iter,
            device=device,
        )

        iter_times = []
        coverages = []
        rewards = []
        collisions = []

        total_transitions = n * steps_per_iter

        t_start = time.time()
        for it in range(1, iterations + 1):
            t0 = time.time()
            metrics = trainer.train_iteration(env, num_steps=steps_per_iter)
            dt = time.time() - t0

            iter_times.append(dt)
            coverages.append(metrics["final_coverage"])
            rewards.append(metrics["mean_reward"])
            collisions.append(metrics["collisions"])

            fps = total_transitions / max(dt, 1e-5)
            print(f"  Iter {it}: {dt:.2f}s | {fps:.1f} steps/s | Coverage: {metrics['final_coverage']*100:.1f}% | Cols: {metrics['collisions']}")

        total_elapsed = time.time() - t_start
        mean_dt = float(np.mean(iter_times))
        overall_fps = float((total_transitions * iterations) / total_elapsed)

        results[f"fleet_{n}"] = {
            "num_drones": n,
            "iterations": iterations,
            "steps_per_iter": steps_per_iter,
            "total_transitions_per_iter": total_transitions,
            "mean_iteration_seconds": mean_dt,
            "effective_fps": overall_fps,
            "final_coverage": float(coverages[-1]),
            "mean_coverage": float(np.mean(coverages)),
            "final_reward": float(rewards[-1]),
            "mean_reward": float(np.mean(rewards)),
            "total_collisions": int(np.sum(collisions)),
            "collisions_per_drone": float(np.sum(collisions) / max(n, 1)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="SwarmRL Fleet Scalability Benchmark")
    parser.add_argument("--fleets", nargs="+", type=int, default=[10, 25, 50], help="Fleet sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Training iterations per fleet size (default: 3)")
    parser.add_argument("--steps", type=int, default=30, help="Steps per iteration (default: 30)")
    parser.add_argument("--output", type=str, default="training/scaling_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("                SwarmRL — MAPPO Fleet Scalability Benchmark                    ")
    print("=" * 80)
    print(f" Fleets: {args.fleets} | Iterations: {args.iterations} | Steps/Iter: {args.steps}")
    print("-" * 80)

    results = benchmark_fleet_scale(
        fleet_sizes=args.fleets,
        iterations=args.iterations,
        steps_per_iter=args.steps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Fleet':>8} | {'Latency':>10} | {'Throughput':>12} | {'Coverage':>10} | {'Cols/Drone':>10}")
    print("-" * 80)
    for k, v in results.items():
        print(f"{v['num_drones']:8d} | {v['mean_iteration_seconds']:8.2f}s | {v['effective_fps']:10.1f} FPS | {v['final_coverage']*100:9.1f}% | {v['collisions_per_drone']:10.2f}")
    print("=" * 80)
    print(f"Detailed scalability report saved to: {args.output}")


if __name__ == "__main__":
    main()
