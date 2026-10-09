"""
SwarmRL -- Day 8: RLlib Training Loop & MAPPO Setup

Registers the PettingZoo SwarmEnv with Ray RLlib and configures the
Multi-Agent PPO (MAPPO) algorithm for decentralized training with a
centralized critic (CTDE).
"""

import os
import sys
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

def get_algorithm_config():
    """
    Configures MAPPO (Multi-Agent PPO).
    In RLlib, PPO with multi-agent setup where agents share policies
    effectively acts as MAPPO.
    """
    env = SwarmEnv()
    
    # We use a single shared policy for all drones (homogeneous swarm)
    # This enables decentralized execution but centralized learning.
    policy_id = "shared_policy"
    
    obs_space = env.observation_space(env.possible_agents[0])
    act_space = env.action_space(env.possible_agents[0])
    
    config = (
        PPOConfig()
        .environment(env="SwarmEnv")
        .framework("torch")
        .env_runners(
            num_env_runners=2,            # Number of parallel environments
            rollout_fragment_length=200,  # Steps per worker before training
        )
        .training(
            train_batch_size=4000,
            minibatch_size=512,
            num_epochs=10,
            lr=5e-5,
            clip_param=0.2,
            vf_clip_param=10.0,
            model={
                "fcnet_hiddens": [256, 256],
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
        .resources(num_gpus=0) # Set to 1 if using GPU
        .debugging(log_level="ERROR")
    )
    
    return config

def main():
    print("=" * 60)
    print("SwarmRL - RLlib MAPPO Training")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the environment
    # RLlib requires PettingZoo environments to be wrapped in PettingZooEnv (or ParallelPettingZooEnv)
    register_env("SwarmEnv", lambda config: ParallelPettingZooEnv(env_creator(config)))
    
    # Get config
    config = get_algorithm_config()
    
    # Create the Algorithm instance
    algo = config.build_algo()
    
    print("Starting training loop...")
    print("Metrics: (Iter) | Mean Reward | Min Reward | Max Reward")
    
    # Train for a few iterations just to test the pipeline
    num_iterations = 5
    
    for i in range(1, num_iterations + 1):
        result = algo.train()
        
        # RLlib accumulates rewards across agents in a multi-agent env, 
        # but 'episode_reward_mean' is typically the sum of all agent rewards.
        # We divide by num_drones to get per-drone average.
        num_drones = WORLD_CONFIG["drones"]["num_drones"]
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0)) / num_drones
        min_reward = result.get('env_runners', {}).get('episode_reward_min', result.get('episode_reward_min', 0)) / num_drones
        max_reward = result.get('env_runners', {}).get('episode_reward_max', result.get('episode_reward_max', 0)) / num_drones
        
        print(f"Iter {i:2d} | Mean: {mean_reward:8.2f} | Min: {min_reward:8.2f} | Max: {max_reward:8.2f}")
    
    # Save the model
    checkpoint_dir = os.path.abspath("checkpoints/day8_mappo")
    algo.save(checkpoint_dir)
    print(f"\nModel saved to: {checkpoint_dir}")
    
    ray.shutdown()
    print("=" * 60)

if __name__ == "__main__":
    main()
