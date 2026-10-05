"""
SwarmRL -- Day 4 Test: Drones Respond to Actions

Validates the complete action -> physics -> collision -> observation loop:
  1. Action scaling: [-1, 1] maps to [-max_accel, max_accel]
  2. Action smoothing: EMA reduces jitter
  3. Single drone responds to directional commands
  4. Full step_all pipeline with multiple drones
  5. Coverage updates during movement
  6. Multi-step simulation (drones fly a trajectory)
"""

import numpy as np
from env.world_config import WORLD_CONFIG, GRID_CELLS_PER_AXIS
from env.drone import Drone
from env.spaces import get_action_space, get_observation_space, OBS_DIM, MAX_ACCEL
from env.action_processor import ActionProcessor

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


def spawn_drones(n=None):
    """Spawn drones at the cluster center."""
    cfg = WORLD_CONFIG["spawn"]
    num = n or WORLD_CONFIG["drones"]["num_drones"]
    center = np.array(cfg["cluster_center"])
    spread = cfg["cluster_spread"]
    drones = []
    for i in range(num):
        pos = center + np.random.uniform(-spread, spread, size=3)
        pos[2] = max(pos[2], cfg["min_altitude"])
        drones.append(Drone(drone_id=i, position=pos))
    return drones


# =====================================================
print("=" * 60)
print("TEST 1: Action Scaling")
print("=" * 60)

proc = ActionProcessor(smoothing=0.0)

# Full forward: [1, 0, 0] -> [max_accel, 0, 0]
scaled = proc.scale_action(np.array([1.0, 0.0, 0.0]))
check("Full +X action scales correctly",
      np.allclose(scaled, [MAX_ACCEL, 0.0, 0.0]),
      f"got {scaled}")

# Full reverse: [-1, -1, -1] -> [-max_accel, -max_accel, -max_accel]
scaled = proc.scale_action(np.array([-1.0, -1.0, -1.0]))
check("Full negative action scales correctly",
      np.allclose(scaled, [-MAX_ACCEL, -MAX_ACCEL, -MAX_ACCEL]),
      f"got {scaled}")

# Zero action
scaled = proc.scale_action(np.array([0.0, 0.0, 0.0]))
check("Zero action stays zero",
      np.allclose(scaled, [0.0, 0.0, 0.0]),
      f"got {scaled}")

# Out-of-range action gets clipped
scaled = proc.scale_action(np.array([5.0, -3.0, 2.0]))
check("Out-of-range action clipped to [-1,1] then scaled",
      np.allclose(scaled, [MAX_ACCEL, -MAX_ACCEL, MAX_ACCEL]),
      f"got {scaled}")

# Half action
scaled = proc.scale_action(np.array([0.5, -0.5, 0.0]))
check("Half action = half acceleration",
      np.allclose(scaled, [MAX_ACCEL * 0.5, -MAX_ACCEL * 0.5, 0.0]),
      f"got {scaled}")


# =====================================================
print("\n" + "=" * 60)
print("TEST 2: Action Smoothing")
print("=" * 60)

proc_smooth = ActionProcessor(smoothing=0.5)

# First action has no smoothing (no history)
a1 = proc_smooth.process_action(0, np.array([1.0, 0.0, 0.0]))
check("First action: no smoothing applied",
      np.allclose(a1, [MAX_ACCEL, 0.0, 0.0]),
      f"got {a1}")

# Second action blends with first (0.5 * prev + 0.5 * current)
a2 = proc_smooth.process_action(0, np.array([0.0, 1.0, 0.0]))
expected = 0.5 * np.array([MAX_ACCEL, 0.0, 0.0]) + 0.5 * np.array([0.0, MAX_ACCEL, 0.0])
check("Second action: smoothed with previous",
      np.allclose(a2, expected, atol=0.01),
      f"expected {expected}, got {a2}")

# No-smoothing processor gives raw values
proc_raw = ActionProcessor(smoothing=0.0)
a1 = proc_raw.process_action(0, np.array([1.0, 0.0, 0.0]))
a2 = proc_raw.process_action(0, np.array([0.0, 1.0, 0.0]))
check("No smoothing: second action is raw",
      np.allclose(a2, [0.0, MAX_ACCEL, 0.0]),
      f"got {a2}")


# =====================================================
print("\n" + "=" * 60)
print("TEST 3: Single Drone Responds to Commands")
print("=" * 60)

proc = ActionProcessor(smoothing=0.0)
drone = Drone(drone_id=0, position=np.array([50.0, 50.0, 25.0]))
dt = WORLD_CONFIG["episode"]["dt"]

# Record initial position
start_pos = drone.position.copy()

# Apply +X acceleration for 10 steps
for _ in range(10):
    proc.step_drone(drone, np.array([1.0, 0.0, 0.0]))

check("Drone moved in +X direction",
      drone.position[0] > start_pos[0],
      f"start={start_pos[0]:.2f}, now={drone.position[0]:.2f}")
check("Drone Y unchanged (approx)",
      abs(drone.position[1] - start_pos[1]) < 0.01,
      f"start={start_pos[1]:.2f}, now={drone.position[1]:.2f}")
check("Drone has positive X velocity",
      drone.velocity[0] > 0,
      f"vel={drone.velocity}")

# Now apply -X to brake
vel_before_brake = drone.velocity[0]
for _ in range(10):
    proc.step_drone(drone, np.array([-1.0, 0.0, 0.0]))

check("Braking reduced X velocity",
      drone.velocity[0] < vel_before_brake,
      f"before={vel_before_brake:.2f}, after={drone.velocity[0]:.2f}")

# Apply upward action
drone_up = Drone(drone_id=1, position=np.array([50.0, 50.0, 10.0]))
start_z = drone_up.position[2]
for _ in range(10):
    proc.step_drone(drone_up, np.array([0.0, 0.0, 1.0]))

check("Drone moved upward",
      drone_up.position[2] > start_z,
      f"start_z={start_z:.2f}, now_z={drone_up.position[2]:.2f}")


# =====================================================
print("\n" + "=" * 60)
print("TEST 4: Full step_all Pipeline")
print("=" * 60)

proc = ActionProcessor(smoothing=0.0)
drones = spawn_drones(5)
coverage_grid = np.zeros(tuple(GRID_CELLS_PER_AXIS), dtype=bool)
obs_space = get_observation_space()

# Create actions: all drones fly in +X
actions = {d.id: np.array([1.0, 0.0, 0.0]) for d in drones}

result = proc.step_all(drones, actions, coverage_grid, current_step=0)

check("Result has 'observations' key", "observations" in result)
check("Result has 'collisions' key", "collisions" in result)
check("Result has 'coverage' key", "coverage" in result)
check("Result has 'new_cells' key", "new_cells" in result)
check("Result has 'applied_accels' key", "applied_accels" in result)

check("Observations for all 5 drones",
      len(result["observations"]) == 5,
      f"got {len(result['observations'])}")

# Each observation should be valid
for did, obs in result["observations"].items():
    in_space = obs_space.contains(obs)
    check(f"Drone {did} observation valid after step",
          in_space and obs.shape == (OBS_DIM,),
          f"shape={obs.shape}, in_space={in_space}")

check("Coverage > 0 after first step",
      result["coverage"] > 0,
      f"coverage={result['coverage']:.4f}")
check("New cells explored > 0",
      result["new_cells"] > 0,
      f"new_cells={result['new_cells']}")

print(f"  (Coverage after 1 step: {result['coverage']:.4f})")
print(f"  (New cells explored: {result['new_cells']})")


# =====================================================
print("\n" + "=" * 60)
print("TEST 5: Multi-Step Simulation (50 steps)")
print("=" * 60)

proc = ActionProcessor(smoothing=0.0)
drones = spawn_drones(10)
coverage_grid = np.zeros(tuple(GRID_CELLS_PER_AXIS), dtype=bool)
act_space = get_action_space()

total_collisions_dd = 0
total_collisions_do = 0
positions_over_time = {d.id: [] for d in drones}

for step in range(50):
    # Each drone gets a random action from the action space
    actions = {d.id: act_space.sample() for d in drones}

    result = proc.step_all(drones, actions, coverage_grid, current_step=step)

    total_collisions_dd += len(result["collisions"]["drone_drone"])
    total_collisions_do += len(result["collisions"]["drone_obstacle"])

    for d in drones:
        positions_over_time[d.id].append(d.position.copy())

# Check that drones actually moved from their spawn positions
moved_count = 0
for d in drones:
    trajectory = np.array(positions_over_time[d.id])
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    if total_distance > 0.1:
        moved_count += 1

check(f"All 10 drones moved during simulation",
      moved_count == 10,
      f"only {moved_count}/10 drones moved")

check("Coverage increased over 50 steps",
      result["coverage"] > 0.01,
      f"coverage={result['coverage']:.4f}")

# Verify all observations are still valid
for did, obs in result["observations"].items():
    assert obs_space.contains(obs), f"Drone {did} obs invalid after 50 steps"
check("All observations valid after 50 steps", True)

print(f"  (Final coverage: {result['coverage']*100:.2f}%)")
print(f"  (Drone-drone collisions: {total_collisions_dd})")
print(f"  (Drone-obstacle collisions: {total_collisions_do})")
print(f"  (Drones that moved: {moved_count}/10)")

# Print sample drone trajectory summary
d0_traj = np.array(positions_over_time[0])
print(f"  (Drone 0 start: {d0_traj[0].round(2)})")
print(f"  (Drone 0 end:   {d0_traj[-1].round(2)})")
print(f"  (Drone 0 total dist: {np.sum(np.linalg.norm(np.diff(d0_traj, axis=0), axis=1)):.2f}m)")


# =====================================================
print("\n" + "=" * 60)
print("TEST 6: Dead Drone Ignores Actions")
print("=" * 60)

proc = ActionProcessor(smoothing=0.0)
dead_drone = Drone(drone_id=99, position=np.array([50.0, 50.0, 25.0]))
dead_drone.alive = False
pos_before = dead_drone.position.copy()

accel = proc.step_drone(dead_drone, np.array([1.0, 1.0, 1.0]))

check("Dead drone returns zero acceleration",
      np.allclose(accel, [0.0, 0.0, 0.0]),
      f"got {accel}")
check("Dead drone position unchanged",
      np.allclose(dead_drone.position, pos_before),
      f"moved to {dead_drone.position}")


# =====================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
if passed_tests == total_tests:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {total_tests - passed_tests} test(s) failed!")
print("=" * 60)
