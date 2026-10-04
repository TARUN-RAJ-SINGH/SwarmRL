"""
SwarmRL -- Day 3 Test: Observation & Action Spaces

Validates:
  1. Gymnasium observation space shape and bounds
  2. Gymnasium action space shape and bounds
  3. LiDAR ray casting against obstacles and world walls
  4. Full observation builder produces correct shape and normalised values
  5. Observations are valid samples of the observation space
"""

import numpy as np
from env.world_config import WORLD_CONFIG, GRID_CELLS_PER_AXIS
from env.drone import Drone
from env.spaces import (
    get_observation_space,
    get_action_space,
    build_observation,
    cast_lidar_rays,
    OBS_DIM,
    ACT_DIM,
    NUM_LIDAR_RAYS,
)

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


# =====================================================
print("=" * 60)
print("TEST 1: Observation Space Definition")
print("=" * 60)

obs_space = get_observation_space()
check("Obs space is Box", obs_space.__class__.__name__ == "Box")
check(f"Obs space shape is ({OBS_DIM},)", obs_space.shape == (OBS_DIM,),
      f"got {obs_space.shape}")
check("Obs space dtype is float32", obs_space.dtype == np.float32)
check("Obs low min >= -1.0", np.all(obs_space.low >= -1.0),
      f"min low = {obs_space.low.min()}")
check("Obs high max <= 1.0", np.all(obs_space.high <= 1.0),
      f"max high = {obs_space.high.max()}")

# Velocity channels should allow [-1, 1]
check("Velocity low = -1.0", np.all(obs_space.low[3:6] == -1.0))
check("Velocity high = 1.0", np.all(obs_space.high[3:6] == 1.0))

# Position channels should be [0, 1]
check("Position low = 0.0", np.all(obs_space.low[0:3] == 0.0))
check("Position high = 1.0", np.all(obs_space.high[0:3] == 1.0))


# =====================================================
print("\n" + "=" * 60)
print("TEST 2: Action Space Definition")
print("=" * 60)

act_space = get_action_space()
check("Act space is Box", act_space.__class__.__name__ == "Box")
check(f"Act space shape is ({ACT_DIM},)", act_space.shape == (ACT_DIM,),
      f"got {act_space.shape}")
check("Act low = -1.0", np.all(act_space.low == -1.0))
check("Act high = 1.0", np.all(act_space.high == 1.0))

# Sample random actions
for _ in range(100):
    a = act_space.sample()
    check_ok = act_space.contains(a)
    if not check_ok:
        check("Random action sample valid", False, f"sample {a}")
        break
else:
    check("100 random action samples all valid", True)


# =====================================================
print("\n" + "=" * 60)
print("TEST 3: LiDAR Ray Casting")
print("=" * 60)

obstacles = WORLD_CONFIG["obstacles"]

# Drone in open space -- all rays should hit far
open_pos = np.array([10.0, 10.0, 25.0], dtype=np.float32)
lidar_open = cast_lidar_rays(open_pos, obstacles)
check(f"LiDAR output shape = ({NUM_LIDAR_RAYS},)", lidar_open.shape == (NUM_LIDAR_RAYS,),
      f"got {lidar_open.shape}")
check("LiDAR values in [0, 1]",
      np.all(lidar_open >= 0.0) and np.all(lidar_open <= 1.0),
      f"range [{lidar_open.min():.3f}, {lidar_open.max():.3f}]")
print(f"  (Open space LiDAR: {lidar_open.round(3)})")

# Drone near an obstacle -- some horizontal rays should detect shorter distances
# Position [25, 45, 10] faces building_1's west face at x=30 (5m away, within sensor_range=10)
near_obs_pos = np.array([25.0, 45.0, 10.0], dtype=np.float32)
lidar_near = cast_lidar_rays(near_obs_pos, obstacles)
# Compare only horizontal rays (first 8) -- vertical always hit ceiling/floor
check("Near-obstacle horizontal LiDAR has shorter readings",
      np.min(lidar_near[:8]) < np.min(lidar_open[:8]),
      f"near horiz min={lidar_near[:8].min():.3f}, open horiz min={lidar_open[:8].min():.3f}")
print(f"  (Near obstacle LiDAR: {lidar_near.round(3)})")


# Drone near world wall -- downward ray should be short
floor_pos = np.array([50.0, 50.0, 1.0], dtype=np.float32)  # near ground
lidar_floor = cast_lidar_rays(floor_pos, obstacles)
down_ray_idx = NUM_LIDAR_RAYS - 1  # last = down
check("Near-floor down-ray is short",
      lidar_floor[down_ray_idx] < 0.15,
      f"down ray = {lidar_floor[down_ray_idx]:.3f}")
print(f"  (Near floor LiDAR: {lidar_floor.round(3)})")


# =====================================================
print("\n" + "=" * 60)
print("TEST 4: Full Observation Builder")
print("=" * 60)

# Spawn drones
num_drones = WORLD_CONFIG["drones"]["num_drones"]
spawn = WORLD_CONFIG["spawn"]
center = np.array(spawn["cluster_center"])
spread = spawn["cluster_spread"]

drones = []
for i in range(num_drones):
    pos = center + np.random.uniform(-spread, spread, size=3)
    pos[2] = max(pos[2], spawn["min_altitude"])
    drones.append(Drone(drone_id=i, position=pos))

# Empty coverage grid
coverage_grid = np.zeros(tuple(GRID_CELLS_PER_AXIS), dtype=bool)

max_steps = WORLD_CONFIG["episode"]["max_steps"]

for drone in drones:
    obs = build_observation(
        drone=drone,
        all_drones=drones,
        obstacles=obstacles,
        coverage_grid=coverage_grid,
        global_coverage_frac=0.0,
        steps_remaining=max_steps,
        max_steps=max_steps,
    )
    check(f"Drone {drone.id} obs shape = ({OBS_DIM},)",
          obs.shape == (OBS_DIM,), f"got {obs.shape}")

    in_space = obs_space.contains(obs)
    check(f"Drone {drone.id} obs is valid sample of obs_space", in_space,
          f"obs = {obs}")

# Print a sample observation breakdown
sample_obs = build_observation(
    drones[0], drones, obstacles, coverage_grid, 0.0, max_steps, max_steps
)
print(f"\n  Sample observation breakdown for Drone 0:")
labels = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "speed_frac", "alt_frac",
    "nearest_drone", "nearest_obs", "neighbours",
    "lidar_0", "lidar_45", "lidar_90", "lidar_135",
    "lidar_180", "lidar_225", "lidar_270", "lidar_315",
    "lidar_up", "lidar_down",
    "local_cov", "global_cov", "wall_prox",
    "steps_remaining", "heading",
]
for label, val in zip(labels, sample_obs):
    print(f"    {label:>18s} = {val:.4f}")


# =====================================================
print("\n" + "=" * 60)
print("TEST 5: Observation with Partial Coverage")
print("=" * 60)

# Mark some cells as explored
coverage_grid[5:15, 5:15, 0:3] = True
explored_cells = int(np.sum(coverage_grid))
total_cells = coverage_grid.size
global_frac = explored_cells / total_cells

obs_cov = build_observation(
    drones[0], drones, obstacles, coverage_grid, global_frac,
    steps_remaining=500, max_steps=max_steps,
)
check("Global coverage > 0 in observation",
      obs_cov[22] > 0.0, f"global_cov = {obs_cov[22]:.4f}")
check("Steps remaining ~0.5",
      abs(obs_cov[24] - 0.5) < 0.01, f"steps_frac = {obs_cov[24]:.4f}")
check("Local coverage > 0 for nearby drone",
      obs_cov[21] > 0.0, f"local_cov = {obs_cov[21]:.4f}")

print(f"  (Explored {explored_cells}/{total_cells} = {global_frac:.4f})")
print(f"  (Observation[22] global_cov = {obs_cov[22]:.4f})")
print(f"  (Observation[21] local_cov  = {obs_cov[21]:.4f})")


# =====================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
if passed_tests == total_tests:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {total_tests - passed_tests} test(s) failed!")
print("=" * 60)
