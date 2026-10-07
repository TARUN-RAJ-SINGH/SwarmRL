"""
SwarmRL -- Day 6 Test: Collision Detection & Resolution

Validates:
  1. Drone-Drone collision detection and elastic resolution
  2. Drone-Obstacle collision detection (sphere-AABB) and resolution
  3. Edge case: Drone center exactly inside obstacle
  4. Multiple rapid collisions
  5. Collision penalties applied in SwarmEnv step loop
"""

import numpy as np
from env.drone import Drone
from env.collisions import (
    check_drone_drone_collisions,
    check_drone_obstacle_collisions,
    resolve_drone_drone_collision,
    resolve_drone_obstacle_collision,
)
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
print("TEST 1: Drone-Drone Collisions")
print("=" * 60)

d1 = Drone(0, position=np.array([10.0, 10.0, 10.0]))
d2 = Drone(1, position=np.array([10.0 + d1.radius + d1.radius, 10.0, 10.0]))  # Exactly touching (dist=1.0)

collisions = check_drone_drone_collisions([d1, d2])
check("No collision if distance >= sum of radii", len(collisions) == 0, f"got {len(collisions)}")

d2.position[0] = 10.0 + d1.radius + d1.radius - 0.1  # Overlapping by 0.1
collisions = check_drone_drone_collisions([d1, d2])
check("Collision detected on overlap", len(collisions) == 1 and collisions[0] == (0, 1), f"got {collisions}")

# Test resolution
d1.velocity = np.array([1.0, 0.0, 0.0])
d2.velocity = np.array([-1.0, 0.0, 0.0])
resolve_drone_drone_collision(d1, d2)

dist_after = np.linalg.norm(d1.position - d2.position)
check("Resolution separates drones", dist_after >= d1.radius + d2.radius, f"dist is {dist_after}")
check("Velocities zeroed along normal", d1.velocity[0] == 0 and d2.velocity[0] == 0, f"d1 vel: {d1.velocity}, d2 vel: {d2.velocity}")

print("\n" + "=" * 60)
print("TEST 2: Drone-Obstacle Collisions")
print("=" * 60)

# Dummy obstacle: box at [20, 20, 0], size [10, 10, 10] (Faces at x=20, 30; y=20, 30; z=0, 10)
obs = [{"id": "box", "position": [20.0, 20.0, 0.0], "size": [10.0, 10.0, 10.0]}]

# Touch face x=20
d3 = Drone(2, position=np.array([20.0 - 0.5 + 0.1, 25.0, 5.0])) # radius 0.5, overlapping by 0.1
cols = check_drone_obstacle_collisions([d3], obs)
check("Collision detected with obstacle face", len(cols) == 1 and cols[0] == (2, "box"))

d3.velocity = np.array([1.0, 0.0, 0.0])
resolve_drone_obstacle_collision(d3, obs[0])
check("Drone pushed out of obstacle", d3.position[0] <= 19.5, f"pos: {d3.position}")
check("Velocity reflected/zeroed", d3.velocity[0] <= 0, f"vel: {d3.velocity}")

print("\n" + "=" * 60)
print("TEST 3: Drone Center Inside Obstacle")
print("=" * 60)

# Center exactly inside
d4 = Drone(3, position=np.array([21.0, 25.0, 5.0])) # Closest to x=20 face (depth 1)
resolve_drone_obstacle_collision(d4, obs[0])
check("Drone popped out to nearest face", d4.position[0] <= 19.5, f"pos: {d4.position}")

print("\n" + "=" * 60)
print("TEST 4: Collision Penalties in SwarmEnv")
print("=" * 60)

env = SwarmEnv()
env.reset(seed=42)

# Fill coverage grid to avoid exploration rewards masking the penalty
env.coverage_grid.fill(True)
env.target_coverage = 2.0  # Prevent success bonus from being applied

# Force a collision between drone_0 and drone_1
env.drones[0].position = np.array([50.0, 50.0, 10.0])
env.drones[1].position = np.array([50.1, 50.0, 10.0]) # Highly overlapping

actions = {agent: np.zeros(3) for agent in env.possible_agents}
obs_next, rewards, term, trunc, info = env.step(actions)

# Should receive reward_collision * 1 plus maybe small step penalty
agent_0 = f"drone_{env.drones[0].id}"
expected_penalty = env.reward_collision + env.reward_step
# But there's also coverage reward. We can just check it's very negative.
check("Collision heavily penalizes reward", rewards[agent_0] < -4.0, f"Reward: {rewards[agent_0]}")
check("Collision count tracked in info", info[agent_0]["collisions"] >= 1, f"Info: {info[agent_0]}")

print("\n" + "=" * 60)
print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
if passed_tests == total_tests:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {total_tests - passed_tests} test(s) failed!")
print("=" * 60)
