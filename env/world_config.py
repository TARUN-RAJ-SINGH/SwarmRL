"""
SwarmRL — Day 2: 3D World & Simulation Configuration

This file defines the complete simulation specification:
  • 3D world dimensions (the disaster zone)
  • Drone properties (count, size, speed, spawn logic)
  • Obstacle definitions (buildings, rubble, walls)
  • Search/target zones
  • Coverage grid resolution

All values are in metres unless noted otherwise.
"""

import numpy as np

# ══════════════════════════════════════════════
# 1.  WORLD DIMENSIONS
# ══════════════════════════════════════════════
WORLD_CONFIG = {
    # --- Bounding Box (metres) ---
    "world": {
        "x_min": 0.0,
        "x_max": 100.0,
        "y_min": 0.0,
        "y_max": 100.0,
        "z_min": 0.0,       # ground level
        "z_max": 50.0,      # max flight altitude
        "description": "100×100×50 m disaster zone",
    },

    # --- Drones ---
    "drones": {
        "num_drones": 10,          # start with 10; will scale to 50
        "radius": 0.5,             # collision sphere radius (m)
        "max_speed": 5.0,          # m/s per axis
        "max_acceleration": 2.5,   # m/s²
        "sensor_range": 10.0,      # observation / LiDAR radius (m)
        "communication_range": 20.0,  # range to detect neighbours
    },

    # --- Spawn Logic ---
    "spawn": {
        "mode": "clustered",       # "random" | "clustered" | "fixed"
        "cluster_center": [10.0, 10.0, 5.0],
        "cluster_spread": 5.0,     # drones spawn within ±spread of center
        "min_altitude": 2.0,       # minimum spawn height
        "description": "Drones deploy from a base-camp corner of the zone",
    },

    # --- Obstacles (static buildings / rubble) ---
    "obstacles": [
        {"id": "building_1", "type": "box",
         "position": [30.0, 40.0, 0.0], "size": [10.0, 10.0, 20.0]},
        {"id": "building_2", "type": "box",
         "position": [60.0, 20.0, 0.0], "size": [8.0, 12.0, 15.0]},
        {"id": "building_3", "type": "box",
         "position": [50.0, 70.0, 0.0], "size": [12.0, 8.0, 25.0]},
        {"id": "rubble_1", "type": "box",
         "position": [75.0, 50.0, 0.0], "size": [6.0, 6.0, 3.0]},
        {"id": "wall_1", "type": "box",
         "position": [20.0, 60.0, 0.0], "size": [2.0, 30.0, 10.0]},
    ],

    # --- Target / Search Zones ---
    "search_zones": [
        {"id": "zone_A", "center": [50.0, 50.0, 0.0], "radius": 45.0,
         "priority": "high", "description": "Primary disaster area"},
        {"id": "zone_B", "center": [85.0, 85.0, 0.0], "radius": 15.0,
         "priority": "medium", "description": "Secondary search sector"},
    ],

    # --- Coverage Grid ---
    "coverage": {
        "grid_resolution": 2.0,     # each cell is 2×2×2 m voxel
        "altitude_layers": 5,       # number of vertical slices for 3D coverage
        "target_coverage": 0.85,    # 85 % = episode success threshold
    },

    # --- Episode Settings ---
    "episode": {
        "max_steps": 1000,
        "dt": 0.1,                  # simulation timestep (seconds)
    },
}


# ══════════════════════════════════════════════
# 2.  DERIVED CONSTANTS (computed from config)
# ══════════════════════════════════════════════
_w = WORLD_CONFIG["world"]
_c = WORLD_CONFIG["coverage"]

WORLD_SIZE = np.array([
    _w["x_max"] - _w["x_min"],
    _w["y_max"] - _w["y_min"],
    _w["z_max"] - _w["z_min"],
])

GRID_CELLS_PER_AXIS = (WORLD_SIZE / _c["grid_resolution"]).astype(int)
TOTAL_GRID_CELLS = int(np.prod(GRID_CELLS_PER_AXIS))

NUM_OBSTACLES = len(WORLD_CONFIG["obstacles"])

print(f"[WorldConfig] World size     : {WORLD_SIZE} m")
print(f"[WorldConfig] Grid per axis  : {GRID_CELLS_PER_AXIS}")
print(f"[WorldConfig] Total voxels   : {TOTAL_GRID_CELLS:,}")
print(f"[WorldConfig] Obstacles      : {NUM_OBSTACLES}")
print(f"[WorldConfig] Drones         : {WORLD_CONFIG['drones']['num_drones']}")
