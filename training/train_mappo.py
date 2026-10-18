"""
SwarmRL -- Day 17: MAPPO CTDE Training Script

Standalone CLI script to train a multi-agent drone swarm
using MAPPO (Centralized Training with Decentralized Execution).

Usage:
    python training/train_mappo.py --drones 10 --iterations 15 --steps 150 --comm
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from env.spaces import OBS_DIM
from env.gnn_comm import ENHANCED_OBS_DIM
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="SwarmRL MAPPO CTDE Training")
    parser.add_argument("--drones", type=int, default=10, help="Number of drones in swarm (default: 10)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of MAPPO iterations (default: 10)")
    parser.add_argument("--steps", type=int, default=150, help="Rollout steps per iteration (default: 150)")
    parser.add_argument("--comm", action="store_true", default=True, help="Use CommSwarmEnv with 42-dim obs")
    parser.add_argument("--no-comm", dest="comm", action="store_false", help="Use standard SwarmEnv (26-dim)")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate (default: 3e-4)")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Critic learning rate (default: 1e-3)")
    parser.add_argument("--ppo-epochs", type=int, default=5, help="PPO epochs per iteration (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Actor mini-batch size (default: 64)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient (default: 0.01)")
    parser.add_argument("--critic-type", type=str, default="attention", choices=["attention", "mlp"], help="Critic architecture")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/day17_mappo", help="Directory for saved models")
    parser.add_argument("--save-interval", type=int, default=5, help="Save interval in iterations (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("                     SwarmRL — Day 17: MAPPO Training                      ")
    print("=" * 80)
    print(f" Fleet Size          : {args.drones} drones")
    print(f" Communication Mode  : {'GNN CommSwarmEnv (42-dim)' if args.comm else 'Base SwarmEnv (26-dim)'}")
    print(f" Critic Architecture : {args.critic_type.upper()} Centralized Critic")
    print(f" Rollout Length      : {args.steps} steps per iteration ({args.drones * args.steps} total transitions/iter)")
    print(f" PPO Optimization    : {args.ppo_epochs} epochs | batch size {args.batch_size} | lr_actor={args.lr_actor}")
    print(f" Value Normalization : PopArt adaptive target rescaling enabled")
    print(f" Device              : {args.device}")
    print("=" * 80)

    # Initialize environment
    obs_dim = ENHANCED_OBS_DIM if args.comm else OBS_DIM
    if args.comm:
        env = CommSwarmEnv(num_drones=args.drones, loss_prob=0.0, formation_bonus=True)
    else:
        env = SwarmEnv(num_drones=args.drones)

    # Configure MAPPO Trainer
    config = MAPPOConfig(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        ppo_epochs=args.ppo_epochs,
        actor_mini_batch_size=args.batch_size,
        critic_mini_batch_size=max(16, args.steps // 5),
        entropy_coef=args.entropy_coef,
        critic_type=args.critic_type,
        use_popart=True,
    )

    trainer = MAPPOTrainer(
        config=config,
        obs_dim=obs_dim,
        num_agents=args.drones,
        rollout_steps=args.steps,
        device=args.device,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\nStarting MAPPO training loop...")
    print("-" * 88)
    print(f"{'Iter':>5} | {'Reward':>8} | {'Cover %':>7} | {'Zone A':>6} | {'Zone B':>6} | {'Cols':>4} | {'Pol Loss':>8} | {'Val Loss':>8} | {'Exp Var':>7} | {'Time':>6}")
    print("-" * 88)

    start_total_time = time.time()

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        metrics = trainer.train_iteration(env, num_steps=args.steps)
        dt = time.time() - t0

        print(
            f"{it:5d} | "
            f"{metrics['mean_reward']:8.3f} | "
            f"{metrics['final_coverage'] * 100:6.1f}% | "
            f"{metrics['zone_a_coverage'] * 100:5.1f}% | "
            f"{metrics['zone_b_coverage'] * 100:5.1f}% | "
            f"{metrics['collisions']:4d} | "
            f"{metrics['policy_loss']:8.4f} | "
            f"{metrics['value_loss']:8.4f} | "
            f"{metrics['explained_variance']:7.3f} | "
            f"{dt:5.2f}s"
        )

        if it % args.save_interval == 0 or it == args.iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, f"mappo_iter_{it}.pt")
            trainer.save_checkpoint(ckpt_path)

    total_time = time.time() - start_total_time
    final_ckpt = os.path.join(args.checkpoint_dir, "day17_mappo_final.pt")
    trainer.save_checkpoint(final_ckpt)

    print("-" * 88)
    print(f"Training completed in {total_time:.2f}s!")
    print(f"Final model checkpoint saved to: {final_ckpt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
