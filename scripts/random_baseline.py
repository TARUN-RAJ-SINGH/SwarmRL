"""
SwarmRL -- Day 7: Random Agent Baseline

Runs a full episode of the SwarmEnv using purely random actions.
Collects metrics (coverage, collisions, rewards) over time and plots them
to establish a baseline performance level before RL training begins.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so we can import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.swarm_env import SwarmEnv
from env.world_config import WORLD_CONFIG

def run_baseline():
    print("=" * 60)
    print("SwarmRL - Random Agent Baseline")
    print("=" * 60)

    env = SwarmEnv()
    obs, infos = env.reset(seed=1337)
    
    max_steps = env.max_steps
    num_drones = WORLD_CONFIG["drones"]["num_drones"]
    
    print(f"Environment initialized with {num_drones} drones.")
    print(f"Max steps: {max_steps}")
    print(f"Target coverage: {env.target_coverage * 100:.1f}%")
    
    # Metrics tracking
    history_coverage = []
    history_collisions = []
    history_rewards = {agent: [] for agent in env.possible_agents}
    
    total_collisions = 0
    step_count = 0
    
    print("\nRunning simulation...")
    
    while env.agents:
        # Sample random actions for all alive agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step the environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track metrics
        agent_0 = list(infos.keys())[0] if infos else env.possible_agents[0]
        
        # Coverage is global, so we can just grab it from any agent's info
        if agent_0 in infos:
            current_coverage = infos[agent_0]["coverage"]
            history_coverage.append(current_coverage)
        else:
            history_coverage.append(history_coverage[-1] if history_coverage else 0.0)
            
        step_collisions = sum(info["collisions"] for info in infos.values())
        total_collisions += step_collisions
        history_collisions.append(step_collisions)
        
        for agent in env.possible_agents:
            history_rewards[agent].append(rewards.get(agent, 0.0))
            
        step_count += 1
        
        if step_count % 100 == 0:
            cov_pct = history_coverage[-1] * 100
            print(f"  Step {step_count}/{max_steps} | Coverage: {cov_pct:.2f}% | Total Collisions: {total_collisions}")

    # Final summary
    final_coverage = history_coverage[-1] * 100 if history_coverage else 0.0
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Steps completed : {step_count}")
    print(f"Final coverage  : {final_coverage:.2f}%")
    print(f"Total collisions: {total_collisions}")
    
    avg_reward = np.mean([sum(history_rewards[a]) for a in env.possible_agents])
    print(f"Avg total reward: {avg_reward:.2f} per drone")
    
    # Plotting
    print("\nGenerating performance plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Coverage over time
    ax1.plot(np.array(history_coverage) * 100, 'b-', linewidth=2)
    ax1.axhline(y=env.target_coverage * 100, color='g', linestyle='--', label='Target Coverage')
    ax1.set_title('Global Coverage over Time')
    ax1.set_ylabel('Coverage (%)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Cumulative Collisions
    ax2.plot(np.cumsum(history_collisions), 'r-', linewidth=2)
    ax2.set_title('Cumulative Collisions')
    ax2.set_ylabel('Total Collisions')
    ax2.grid(True)
    
    # Plot 3: Average Cumulative Reward
    avg_cum_reward = np.mean([np.cumsum(history_rewards[a]) for a in env.possible_agents], axis=0)
    ax3.plot(avg_cum_reward, 'k-', linewidth=2)
    ax3.set_title('Average Cumulative Reward per Drone')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'baseline_metrics.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_baseline()
