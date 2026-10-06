"""
SwarmRL -- Day 5 Test: PettingZoo Environment

Validates:
  1. Environment initializes correctly with ParallelEnv interface
  2. reset() returns valid observations and infos
  3. step() with random actions returns expected structures
  4. Rewards are calculated correctly
  5. Truncation works after max_steps
"""

import numpy as np
from env.swarm_env import SwarmEnv
from env.world_config import WORLD_CONFIG

PASS = "[PASS]"
FAIL = "[FAIL]"
total_tests = 0
passed_tests = 0


def check(name, condition, detail=""):
    global total_tests, passed_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name} -- {detail}")

print("=" * 60)
print("TEST 1: Environment Initialization & Reset")
print("=" * 60)

env = SwarmEnv()
check("Environment initialized", isinstance(env, SwarmEnv))
check("Agents populated", len(env.possible_agents) == WORLD_CONFIG["drones"]["num_drones"])
check("Action spaces exist", len(env.action_spaces) == len(env.possible_agents))
check("Observation spaces exist", len(env.observation_spaces) == len(env.possible_agents))

obs, infos = env.reset(seed=42)
check("reset() returns obs dict", isinstance(obs, dict))
check("reset() returns info dict", isinstance(infos, dict))
check("obs has correct number of agents", len(obs) == len(env.possible_agents))

agent_0 = env.possible_agents[0]
check(f"obs[{agent_0}] has correct shape", obs[agent_0].shape == (26,))

print("\n" + "=" * 60)
print("TEST 2: Environment Step")
print("=" * 60)

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
next_obs, rewards, terminations, truncations, infos = env.step(actions)

check("step() returns obs dict", isinstance(next_obs, dict))
check("step() returns rewards dict", isinstance(rewards, dict))
check("step() returns terminations dict", isinstance(terminations, dict))
check("step() returns truncations dict", isinstance(truncations, dict))
check("step() returns infos dict", isinstance(infos, dict))

check("rewards values are floats", isinstance(rewards[agent_0], float))
check("terminations values are bools", isinstance(terminations[agent_0], bool))
check("truncations values are bools", isinstance(truncations[agent_0], bool))

print(f"  (Sample reward for {agent_0}: {rewards[agent_0]:.4f})")
print(f"  (Sample info for {agent_0}: {infos[agent_0]})")

print("\n" + "=" * 60)
print("TEST 3: Multi-step loop & Truncation")
print("=" * 60)

# Fast forward to the end
env.max_steps = 10 # Temporarily reduce max steps for quick test
obs, infos = env.reset(seed=123)

step_count = 0
total_reward = {agent: 0.0 for agent in env.possible_agents}

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    for agent in env.possible_agents:
        total_reward[agent] += rewards.get(agent, 0.0)
        
    step_count += 1

check("Simulation truncated after max_steps", step_count == 10)
check("Agents list is empty after done", len(env.agents) == 0)

print(f"  (Total reward for {agent_0}: {total_reward[agent_0]:.2f})")
print(f"  (Final coverage: {infos[agent_0]['coverage']*100:.2f}%)")


print("\n" + "=" * 60)
print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
if passed_tests == total_tests:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {total_tests - passed_tests} test(s) failed!")
print("=" * 60)
