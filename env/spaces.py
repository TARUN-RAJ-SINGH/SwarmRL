"""
SwarmRL -- Day 3: Observation & Action Space Definitions

Defines formal Gymnasium spaces for the multi-agent drone environment:

Observation Space (per drone) -- 26 dimensions:
  ---------------------------------------------------------------
  Index   | Feature                         | Range (normalised)
  ---------------------------------------------------------------
  0-2     | Position (x, y, z)              | [0, 1]
  3-5     | Velocity (vx, vy, vz)           | [-1, 1]
  6       | Speed fraction (|v| / max_speed) | [0, 1]
  7       | Altitude fraction (z / z_max)   | [0, 1]
  8       | Nearest drone distance          | [0, 1]  (clamped to sensor_range)
  9       | Nearest obstacle distance       | [0, 1]  (clamped to sensor_range)
  10      | Neighbour count (normalised)    | [0, 1]
  11-18   | LiDAR rays (8 horizontal dirs)  | [0, 1]  (fraction of sensor_range)
  19-20   | LiDAR rays (up / down)          | [0, 1]
  21      | Local coverage fraction         | [0, 1]  (explored cells nearby)
  22      | Global coverage fraction        | [0, 1]
  23      | Wall proximity (any axis)       | [0, 1]
  24      | Steps remaining fraction        | [0, 1]
  25      | Heading angle (normalised)      | [0, 1]  (atan2(vy, vx) -> [0, 1])
  ---------------------------------------------------------------

Action Space (per drone) -- 3 dimensions:
  [ax, ay, az]  continuous acceleration in [-1, 1], scaled by max_acceleration
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.world_config import WORLD_CONFIG, WORLD_SIZE, GRID_CELLS_PER_AXIS


# ======================================================
# Constants derived from config
# ======================================================
_drone_cfg = WORLD_CONFIG["drones"]
_world_cfg = WORLD_CONFIG["world"]
_episode_cfg = WORLD_CONFIG["episode"]

NUM_DRONES = _drone_cfg["num_drones"]
MAX_SPEED = _drone_cfg["max_speed"]
MAX_ACCEL = _drone_cfg["max_acceleration"]
SENSOR_RANGE = _drone_cfg["sensor_range"]
COMM_RANGE = _drone_cfg["communication_range"]

WORLD_ORIGIN = np.array([_world_cfg["x_min"], _world_cfg["y_min"], _world_cfg["z_min"]])
WORLD_EXTENT = WORLD_SIZE  # [100, 100, 50]

NUM_LIDAR_HORIZONTAL = 8   # rays every 45 deg in the XY plane
NUM_LIDAR_VERTICAL = 2     # up + down
NUM_LIDAR_RAYS = NUM_LIDAR_HORIZONTAL + NUM_LIDAR_VERTICAL

OBS_DIM = 26
ACT_DIM = 3

# Pre-compute horizontal LiDAR ray directions (unit vectors in XY, z=0)
LIDAR_HORIZONTAL_DIRS = np.array([
    [np.cos(angle), np.sin(angle), 0.0]
    for angle in np.linspace(0, 2 * np.pi, NUM_LIDAR_HORIZONTAL, endpoint=False)
], dtype=np.float32)

# Vertical LiDAR: straight up and straight down
LIDAR_VERTICAL_DIRS = np.array([
    [0.0, 0.0, 1.0],    # up
    [0.0, 0.0, -1.0],   # down
], dtype=np.float32)


# ======================================================
# Space Builders
# ======================================================

def get_observation_space() -> spaces.Box:
    """
    Returns a Gymnasium Box representing the normalised observation space
    for a single drone. All values are in [0, 1] or [-1, 1].
    """
    low = np.array([
        # position xyz normalised        (3)
        0.0, 0.0, 0.0,
        # velocity xyz normalised         (3)
        -1.0, -1.0, -1.0,
        # speed fraction                  (1)
        0.0,
        # altitude fraction               (1)
        0.0,
        # nearest drone dist normalised   (1)
        0.0,
        # nearest obstacle dist norm.     (1)
        0.0,
        # neighbour count normalised      (1)
        0.0,
        # LiDAR horizontal (8)
        *([0.0] * NUM_LIDAR_HORIZONTAL),
        # LiDAR vertical (2)
        *([0.0] * NUM_LIDAR_VERTICAL),
        # local coverage fraction         (1)
        0.0,
        # global coverage fraction        (1)
        0.0,
        # wall proximity                  (1)
        0.0,
        # steps remaining fraction        (1)
        0.0,
        # heading angle normalised        (1)
        0.0,
    ], dtype=np.float32)

    high = np.ones_like(low, dtype=np.float32)
    # velocity can be negative
    high[3:6] = 1.0
    low[3:6] = -1.0

    assert low.shape[0] == OBS_DIM, f"Low shape {low.shape[0]} != OBS_DIM {OBS_DIM}"
    assert high.shape[0] == OBS_DIM, f"High shape {high.shape[0]} != OBS_DIM {OBS_DIM}"

    return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


def get_action_space() -> spaces.Box:
    """
    Returns a Gymnasium Box representing the continuous action space.
    Actions are normalised to [-1, 1] and scaled by max_acceleration
    inside the environment step.
    """
    return spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(ACT_DIM,),
        dtype=np.float32,
    )


# ======================================================
# LiDAR Ray Casting
# ======================================================

def _ray_aabb_distance(origin: np.ndarray, direction: np.ndarray,
                       box_min: np.ndarray, box_max: np.ndarray,
                       max_dist: float) -> float:
    """
    Cast a ray from `origin` in `direction` and return the distance to
    the nearest AABB face, or `max_dist` if no hit within range.
    Uses the slab method for ray-box intersection.
    """
    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        if abs(direction[i]) < 1e-8:
            # Ray is parallel to this slab
            if origin[i] < box_min[i] or origin[i] > box_max[i]:
                return max_dist  # outside the slab, no hit
            # else: inside this slab, t range is (-inf, +inf) — no constraint
        else:
            t1 = (box_min[i] - origin[i]) / direction[i]
            t2 = (box_max[i] - origin[i]) / direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            if t_near > t_far or t_far < 0:
                return max_dist  # no hit

    hit_dist = t_near if t_near >= 0 else t_far
    return min(float(hit_dist), max_dist)


def cast_lidar_rays(position: np.ndarray, obstacles: list) -> np.ndarray:
    """
    Cast 10 LiDAR rays (8 horizontal + 2 vertical) from the drone's
    position and return normalised distances [0, 1].

    Also considers world boundaries as solid walls.

    Returns:
        np.ndarray of shape (10,) with values in [0, 1]
    """
    all_dirs = np.concatenate([LIDAR_HORIZONTAL_DIRS, LIDAR_VERTICAL_DIRS], axis=0)
    distances = np.ones(NUM_LIDAR_RAYS, dtype=np.float32)  # default = 1.0 (max range)

    for i, direction in enumerate(all_dirs):
        min_hit = SENSOR_RANGE

        # Check obstacles
        for obs in obstacles:
            box_min = np.array(obs["position"], dtype=np.float32)
            box_max = box_min + np.array(obs["size"], dtype=np.float32)
            hit = _ray_aabb_distance(position, direction, box_min, box_max, SENSOR_RANGE)
            min_hit = min(min_hit, hit)

        # Check world boundaries (6 planes)
        world_min = WORLD_ORIGIN.astype(np.float32)
        world_max = (WORLD_ORIGIN + WORLD_EXTENT).astype(np.float32)

        for axis in range(3):
            if abs(direction[axis]) > 1e-8:
                # Distance to min boundary along this axis
                t_low = (world_min[axis] - position[axis]) / direction[axis]
                if t_low > 0:
                    min_hit = min(min_hit, t_low)
                # Distance to max boundary along this axis
                t_high = (world_max[axis] - position[axis]) / direction[axis]
                if t_high > 0:
                    min_hit = min(min_hit, t_high)

        distances[i] = float(np.clip(min_hit / SENSOR_RANGE, 0.0, 1.0))

    return distances


# ======================================================
# Observation Builder
# ======================================================

def build_observation(
    drone,
    all_drones: list,
    obstacles: list,
    coverage_grid: np.ndarray,
    global_coverage_frac: float,
    steps_remaining: int,
    max_steps: int,
) -> np.ndarray:
    """
    Build a fully normalised 26-dim observation vector for a single drone.

    Args:
        drone:               The Drone instance
        all_drones:          List of all Drone instances
        obstacles:           List of obstacle dicts from WORLD_CONFIG
        coverage_grid:       3D numpy bool array of explored voxels
        global_coverage_frac: Current fraction of total cells explored [0, 1]
        steps_remaining:     Steps left in the episode
        max_steps:           Max steps per episode

    Returns:
        np.ndarray of shape (OBS_DIM,) with dtype float32
    """
    pos = drone.position
    vel = drone.velocity

    # --- Normalised position [0, 1] ---
    norm_pos = (pos - WORLD_ORIGIN) / WORLD_EXTENT

    # --- Normalised velocity [-1, 1] ---
    norm_vel = vel / MAX_SPEED

    # --- Speed fraction [0, 1] ---
    speed_frac = np.clip(np.linalg.norm(vel) / MAX_SPEED, 0.0, 1.0)

    # --- Altitude fraction [0, 1] ---
    alt_frac = norm_pos[2]

    # --- Nearest drone distance [0, 1] ---
    nearest_drone = SENSOR_RANGE
    neighbour_count = 0
    for other in all_drones:
        if other.id == drone.id or not other.alive:
            continue
        dist = np.linalg.norm(pos - other.position)
        nearest_drone = min(nearest_drone, dist)
        if dist <= COMM_RANGE:
            neighbour_count += 1
    norm_nearest_drone = np.clip(nearest_drone / SENSOR_RANGE, 0.0, 1.0)

    # --- Nearest obstacle distance [0, 1] ---
    nearest_obs = SENSOR_RANGE
    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_size = np.array(obs["size"])
        clamped = np.clip(pos, obs_pos, obs_pos + obs_size)
        dist = np.linalg.norm(pos - clamped)
        nearest_obs = min(nearest_obs, dist)
    norm_nearest_obs = np.clip(nearest_obs / SENSOR_RANGE, 0.0, 1.0)

    # --- Neighbour count normalised [0, 1] ---
    max_neighbours = max(NUM_DRONES - 1, 1)
    norm_neighbours = np.clip(neighbour_count / max_neighbours, 0.0, 1.0)

    # --- LiDAR rays (10 values) ---
    lidar = cast_lidar_rays(pos.astype(np.float32), obstacles)

    # --- Local coverage fraction [0, 1] ---
    local_cov = _compute_local_coverage(pos, coverage_grid)

    # --- Global coverage fraction [0, 1] ---
    global_cov = np.clip(global_coverage_frac, 0.0, 1.0)

    # --- Wall proximity [0, 1] (1 = touching wall) ---
    wall_prox = _compute_wall_proximity(pos)

    # --- Steps remaining fraction [0, 1] ---
    steps_frac = np.clip(steps_remaining / max(max_steps, 1), 0.0, 1.0)

    # --- Heading angle normalised [0, 1] ---
    heading = np.arctan2(vel[1], vel[0])  # [-pi, pi]
    norm_heading = (heading + np.pi) / (2 * np.pi)  # [0, 1]

    obs = np.array([
        *norm_pos,            # 0-2
        *norm_vel,            # 3-5
        speed_frac,           # 6
        alt_frac,             # 7
        norm_nearest_drone,   # 8
        norm_nearest_obs,     # 9
        norm_neighbours,      # 10
        *lidar,               # 11-20  (8 horizontal + 2 vertical)
        local_cov,            # 21
        global_cov,           # 22
        wall_prox,            # 23
        steps_frac,           # 24
        norm_heading,         # 25
    ], dtype=np.float32)

    assert obs.shape == (OBS_DIM,), f"Obs shape {obs.shape} != ({OBS_DIM},)"
    return obs


# ======================================================
# Helper Functions
# ======================================================

def _compute_local_coverage(position: np.ndarray, coverage_grid: np.ndarray) -> float:
    """
    Compute what fraction of voxels within a local neighbourhood (sensor_range)
    around the drone have already been explored.
    Returns a value in [0, 1].
    """
    res = WORLD_CONFIG["coverage"]["grid_resolution"]

    # Convert position to grid indices
    grid_pos = ((position - WORLD_ORIGIN) / res).astype(int)

    # Local neighbourhood radius in grid cells
    radius_cells = int(np.ceil(SENSOR_RANGE / res))

    # Compute bounds
    lo = np.maximum(grid_pos - radius_cells, 0)
    hi = np.minimum(grid_pos + radius_cells + 1, GRID_CELLS_PER_AXIS)

    if np.any(hi <= lo):
        return 0.0

    local_slice = coverage_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    total_cells = max(local_slice.size, 1)
    explored = np.sum(local_slice)

    return float(explored / total_cells)


def _compute_wall_proximity(position: np.ndarray) -> float:
    """
    Returns a value in [0, 1] where 1.0 means touching a world boundary
    and 0.0 means at least sensor_range away from all walls.
    """
    dist_to_walls = np.array([
        position[0] - _world_cfg["x_min"],
        _world_cfg["x_max"] - position[0],
        position[1] - _world_cfg["y_min"],
        _world_cfg["y_max"] - position[1],
        position[2] - _world_cfg["z_min"],
        _world_cfg["z_max"] - position[2],
    ])
    min_wall_dist = max(np.min(dist_to_walls), 0.0)
    return float(np.clip(1.0 - min_wall_dist / SENSOR_RANGE, 0.0, 1.0))
