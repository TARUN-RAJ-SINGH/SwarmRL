"""
SwarmRL -- Day 15: GNN MAPPO Training

Trains MAPPO over the CommSwarmEnv that augments each drone's 26-dim base
observation with a 16-dim inter-agent communication context (42-dim total).

Changes from Day 9 (train.py):
  - Registers CommSwarmEnv instead of SwarmEnv
  - Observation dim: 42 (was 26)
  - Deeper network: [512, 256, 256] (was [256, 256, 256])
  - Larger batch: train_batch_size=10000 for cooperative learning
  - Entropy coeff=0.01 to encourage exploration diversity
  - Checkpoint tag: "day15_gnn_mappo_iter_{i}"

Usage:
  python training/train_gnn.py --iterations 50
  python training/train_gnn.py --iterations 200 --drones 25
  python training/train_gnn.py --iterations 100 --loss-prob 0.2
"""

import os
import sys
import argparse
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.comm_swarm_env import CommSwarmEnv
from env.gnn_comm       import ENHANCED_OBS_DIM
from env.world_config   import WORLD_CONFIG


# ─────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────

def env_creator(env_config: dict):
    """Creates CommSwarmEnv with configurable parameters."""
    return CommSwarmEnv(
        num_drones    = env_config.get("num_drones",    None),
        loss_prob     = env_config.get("loss_prob",     0.0),
        max_msgs      = env_config.get("max_msgs",      9),
        formation_bonus = env_config.get("formation_bonus", True),
    )


# ─────────────────────────────────────────────────────────────
# Algorithm config
# ─────────────────────────────────────────────────────────────

def get_gnn_config(log_dir: str, args) -> PPOConfig:
    """
    Configures MAPPO for cooperative GNN-augmented communication training.

    Key differences from Day 9:
      - Observation dim: 42 (26 base + 16 comm context)
      - Wider + deeper network for cross-agent feature learning
      - Entropy coeff encourages diverse exploration patterns
      - Higher batch size to capture multi-agent correlations
    """
    # Build a sample env to extract spaces
    sample_env = CommSwarmEnv(
        num_drones    = args.drones,
        loss_prob     = args.loss_prob,
        formation_bonus = True,
    )
    policy_id  = "gnn_shared_policy"
    obs_space  = sample_env.observation_space(sample_env.possible_agents[0])
    act_space  = sample_env.action_space(sample_env.possible_agents[0])

    env_config = {
        "num_drones":     args.drones,
        "loss_prob":      args.loss_prob,
        "max_msgs":       args.max_msgs,
        "formation_bonus": True,
    }

    config = (
        PPOConfig()
        .environment(
            env           = "CommSwarmEnv",
            env_config    = env_config,
        )
        .framework("torch")
        .env_runners(
            num_env_runners       = 2,
            rollout_fragment_length = 500,
        )
        .training(
            train_batch_size = 10000,          # larger for cooperative learning
            minibatch_size   = 1024,
            num_epochs       = 10,
            lr               = 5e-5,           # conservative for stable comm learning
            clip_param       = 0.2,
            vf_clip_param    = 20.0,
            entropy_coeff    = 0.01,           # encourage exploration diversity
            model={
                # Wider to capture 42-dim comm-augmented observations
                "fcnet_hiddens":    [512, 256, 256],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies={
                policy_id: (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: policy_id,
        )
        .debugging(log_level="ERROR")
        .resources(num_gpus=0)
    )

    # TensorBoard logging
    config.logger_config = {
        "type": "tensorboard",
        "logdir": log_dir,
    }

    return config


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    print("=" * 65)
    print("SwarmRL — Day 15: GNN MAPPO Training (CommSwarmEnv)")
    print("=" * 65)
    print(f"  Fleet size  : {args.drones} drones")
    print(f"  Loss prob   : {args.loss_prob:.2%}")
    print(f"  Max msgs    : {args.max_msgs}")
    print(f"  Obs dim     : {ENHANCED_OBS_DIM} (26 base + 16 comm context)")
    print(f"  Iterations  : {args.iterations}")
    print("=" * 65)

    ray.init(ignore_reinit_error=True)

    register_env(
        "CommSwarmEnv",
        lambda cfg: ParallelPettingZooEnv(env_creator(cfg)),
    )

    log_dir = os.path.abspath("training/logs/gnn_mappo_run")
    os.makedirs(log_dir, exist_ok=True)

    config = get_gnn_config(log_dir, args)
    algo   = config.build_algo()

    print(f"TensorBoard logs → {log_dir}")
    print("Iter  | Mean Reward | Min Reward | Max Reward")
    print("-" * 50)

    num_drones = args.drones or WORLD_CONFIG["drones"]["num_drones"]

    for i in range(1, args.iterations + 1):
        result       = algo.train()
        env_runners  = result.get("env_runners", {})
        mean_r = env_runners.get("episode_reward_mean",
                  result.get("episode_reward_mean", 0.0)) / num_drones
        min_r  = env_runners.get("episode_reward_min",
                  result.get("episode_reward_min",  0.0)) / num_drones
        max_r  = env_runners.get("episode_reward_max",
                  result.get("episode_reward_max",  0.0)) / num_drones

        print(f" {i:3d}  | {mean_r:11.2f} | {min_r:10.2f} | {max_r:10.2f}")

        if i % 10 == 0 or i == args.iterations:
            ckpt_dir = os.path.abspath(f"checkpoints/day15_gnn_mappo_iter_{i}")
            algo.save(ckpt_dir)
            print(f"  → Checkpoint saved: {ckpt_dir}")

    ray.shutdown()
    print("=" * 65)
    print("GNN MAPPO training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL GNN MAPPO Training")
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of training iterations (default: 10)",
    )
    parser.add_argument(
        "--drones", type=int, default=None,
        help="Fleet size override (default: from world_config)",
    )
    parser.add_argument(
        "--loss-prob", type=float, default=0.0,
        dest="loss_prob",
        help="Packet-loss probability per comm link [0,1] (default: 0.0)",
    )
    parser.add_argument(
        "--max-msgs", type=int, default=9,
        dest="max_msgs",
        help="Bandwidth cap: max neighbours aggregated per step (default: 9)",
    )
    args = parser.parse_args()
    main(args)
