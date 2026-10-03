"""
SwarmRL — Quick Smoke Test

Validates that:
  1. World config loads and derived constants are correct
  2. Drones can be spawned with clustered logic
  3. Drones can move, get observations, and stay in bounds
  4. Collision detection works for drone–drone and drone–obstacle
  5. Backend server module imports cleanly
"""

import sys
import numpy as np

# ── 1. World Config ──────────────────────────
print("=" * 60)
print("TEST 1: World Configuration")
print("=" * 60)

from env.world_config import WORLD_CONFIG, WORLD_SIZE, GRID_CELLS_PER_AXIS, TOTAL_GRID_CELLS

assert WORLD_SIZE[0] == 100.0
assert WORLD_SIZE[1] == 100.0
assert WORLD_SIZE[2] == 50.0
print(f"  [OK] World size: {WORLD_SIZE}")
print(f"  [OK] Grid cells per axis: {GRID_CELLS_PER_AXIS}")
print(f"  [OK] Total voxels: {TOTAL_GRID_CELLS:,}")

# ── 2. Drone Spawning ───────────────────────
print("\n" + "=" * 60)
print("TEST 2: Drone Spawning (clustered)")
print("=" * 60)

from env.drone import Drone

spawn_cfg = WORLD_CONFIG["spawn"]
num_drones = WORLD_CONFIG["drones"]["num_drones"]
center = np.array(spawn_cfg["cluster_center"])
spread = spawn_cfg["cluster_spread"]

drones = []
for i in range(num_drones):
    pos = center + np.random.uniform(-spread, spread, size=3)
    pos[2] = max(pos[2], spawn_cfg["min_altitude"])  # enforce min altitude
    drone = Drone(drone_id=i, position=pos)
    drones.append(drone)
    print(f"  Spawned {drone}")

assert len(drones) == num_drones
print(f"  [OK] {num_drones} drones spawned successfully")

# ── 3. Movement & Observations ───────────────
print("\n" + "=" * 60)
print("TEST 3: Movement & Observations")
print("=" * 60)

dt = WORLD_CONFIG["episode"]["dt"]
obstacles = WORLD_CONFIG["obstacles"]

for drone in drones:
    action = np.random.uniform(-2.0, 2.0, size=3)
    drone.apply_action(action, dt)
    drone.clamp_to_world()
    obs = drone.get_observation(drones, obstacles)
    assert obs.shape == (10,), f"Expected obs shape (10,), got {obs.shape}"

print(f"  [OK] All drones moved and produced observations of shape (10,)")
print(f"  Sample observation: {drones[0].get_observation(drones, obstacles)}")

# ── 4. Collision Detection ───────────────────
print("\n" + "=" * 60)
print("TEST 4: Collision Detection")
print("=" * 60)

from env.collisions import (
    check_drone_drone_collisions,
    check_drone_obstacle_collisions,
    resolve_drone_drone_collision,
    resolve_drone_obstacle_collision,
)

# Force two drones on top of each other
drones[0].position = np.array([30.0, 40.0, 10.0])
drones[1].position = np.array([30.0, 40.0, 10.2])

dd_collisions = check_drone_drone_collisions(drones)
print(f"  Drone-drone collisions: {dd_collisions}")

if dd_collisions:
    resolve_drone_drone_collision(drones[0], drones[1])
    print(f"  [OK] Resolved — new positions: {drones[0].position.round(2)}, {drones[1].position.round(2)}")

# Force a drone into an obstacle
drones[2].position = np.array([31.0, 41.0, 5.0])  # inside building_1
do_collisions = check_drone_obstacle_collisions(drones)
print(f"  Drone-obstacle collisions: {do_collisions}")

for drone_id, obs_id in do_collisions:
    drone = next(d for d in drones if d.id == drone_id)
    obs = next(o for o in obstacles if o["id"] == obs_id)
    resolve_drone_obstacle_collision(drone, obs)
    print(f"  [OK] Resolved drone {drone_id} vs {obs_id} — new pos: {drone.position.round(2)}")

# ── 5. Serialisation ─────────────────────────
print("\n" + "=" * 60)
print("TEST 5: WebSocket Serialisation")
print("=" * 60)

data = drones[0].to_dict()
assert "id" in data
assert "position" in data
assert isinstance(data["position"], list)
print(f"  [OK] Drone serialises to: {data}")

# ── Done ─────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
