"""
SwarmRL -- Day 9: TensorBoard Integration & Hyperparameter Tuning

Registers the PettingZoo SwarmEnv with Ray RLlib, configures MAPPO,
adds TensorBoard logging, and tunes hyperparameters for better coverage.
"""

import os
import sys
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Add project root to path so we can import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.swarm_env import SwarmEnv
from env.world_config import WORLD_CONFIG

def env_creator(env_config):
    """Creates and returns the PettingZoo SwarmEnv."""
    return SwarmEnv()

def get_algorithm_config(log_dir):
    """
    Configures MAPPO (Multi-Agent PPO) with TensorBoard logging.
    """
    env = SwarmEnv()
    policy_id = "shared_policy"
    
    obs_space = env.observation_space(env.possible_agents[0])
    act_space = env.action_space(env.possible_agents[0])
    
    config = (
        PPOConfig()
        .environment(env="SwarmEnv")
        .framework("torch")
        .env_runners(
            num_env_runners=2,            # Number of parallel environments
            rollout_fragment_length=500,  # Longer fragments for better advantage estimation
        )
        .training(
            train_batch_size=8000,        # Larger batch size for stable multi-agent gradients
            minibatch_size=1024,
            num_epochs=10,
            lr=1e-4,                      # Slightly higher LR for initial exploration
            clip_param=0.2,
            vf_clip_param=20.0,           # Allow larger value function updates
            model={
                "fcnet_hiddens": [256, 256, 256], # Deeper network for spatial reasoning
                "fcnet_activation": "relu",
            }
        )
        .multi_agent(
            policies={
                policy_id: (None, obs_space, act_space, {}),
            },
            # Map all drone agents to the same shared policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: policy_id,
        )
        .debugging(log_level="ERROR")
        .resources(num_gpus=0)
    )
    
    # Configure TensorBoard logger directory
    config.logger_config = {
        "type": "tensorboard",
        "logdir": log_dir,
    }
    
    return config

def main(args):
    print("=" * 60)
    print("SwarmRL - RLlib MAPPO Training")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the environment
    register_env("SwarmEnv", lambda config: ParallelPettingZooEnv(env_creator(config)))
    
    # Setup TensorBoard log directory
    log_dir = os.path.abspath("training/logs/mappo_run")
    os.makedirs(log_dir, exist_ok=True)
    
    # Get config
    config = get_algorithm_config(log_dir)
    algo = config.build_algo()
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Starting training loop for {args.iterations} iterations...")
    print("Metrics: (Iter) | Mean Reward | Min Reward | Max Reward")
    
    for i in range(1, args.iterations + 1):
        result = algo.train()
        
        # Calculate per-drone rewards
        num_drones = WORLD_CONFIG["drones"]["num_drones"]
        env_runners = result.get('env_runners', {})
        mean_reward = env_runners.get('episode_reward_mean', result.get('episode_reward_mean', 0)) / num_drones
        min_reward = env_runners.get('episode_reward_min', result.get('episode_reward_min', 0)) / num_drones
        max_reward = env_runners.get('episode_reward_max', result.get('episode_reward_max', 0)) / num_drones
        
        print(f"Iter {i:3d} | Mean: {mean_reward:8.2f} | Min: {min_reward:8.2f} | Max: {max_reward:8.2f}")
        
        # Save checkpoints periodically
        if i % 10 == 0 or i == args.iterations:
            checkpoint_dir = os.path.abspath(f"checkpoints/day9_mappo_iter_{i}")
            algo.save(checkpoint_dir)
            print(f"  -> Checkpoint saved at Iter {i}")
            
    ray.shutdown()
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL MAPPO Training")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    args = parser.parse_args()
    main(args)
