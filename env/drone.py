"""
SwarmRL — Drone Agent Model

Represents a single drone in the simulation with its physical state,
sensor readings, and movement logic. Used by the PettingZoo environment.
"""

import numpy as np
from env.world_config import WORLD_CONFIG


class Drone:
    """A single autonomous drone agent."""

    def __init__(self, drone_id: int, position: np.ndarray = None):
        cfg = WORLD_CONFIG["drones"]

        self.id = drone_id
        self.radius = cfg["radius"]
        self.max_speed = cfg["max_speed"]
        self.max_acceleration = cfg["max_acceleration"]
        self.sensor_range = cfg["sensor_range"]
        self.communication_range = cfg["communication_range"]

        # Physical state
        self.position = position if position is not None else np.zeros(3)
        self.velocity = np.zeros(3)

        # Status
        self.alive = True
        self.collisions = 0
        self.cells_explored = 0

    # ──────────────────────────────────────────
    # Movement
    # ──────────────────────────────────────────

    def apply_action(self, action: np.ndarray, dt: float):
        """
        Apply a continuous action [ax, ay, az] (acceleration) to update
        the drone's velocity and position.

        Args:
            action: np.array of shape (3,) — acceleration in m/s²
            dt: simulation timestep in seconds
        """
        # Clip acceleration
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        # Update velocity
        self.velocity += action * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # Update position
        self.position += self.velocity * dt

    def clamp_to_world(self):
        """Keep the drone within world boundaries."""
        w = WORLD_CONFIG["world"]
        r = self.radius

        self.position[0] = np.clip(self.position[0], w["x_min"] + r, w["x_max"] - r)
        self.position[1] = np.clip(self.position[1], w["y_min"] + r, w["y_max"] - r)
        self.position[2] = np.clip(self.position[2], w["z_min"] + r, w["z_max"] - r)

        # Zero out velocity on clamped axes
        for i, (lo, hi) in enumerate([
            (w["x_min"] + r, w["x_max"] - r),
            (w["y_min"] + r, w["y_max"] - r),
            (w["z_min"] + r, w["z_max"] - r),
        ]):
            if self.position[i] <= lo or self.position[i] >= hi:
                self.velocity[i] = 0.0

    # ──────────────────────────────────────────
    # Observations
    # ──────────────────────────────────────────

    def get_observation(self, all_drones: list, obstacles: list) -> np.ndarray:
        """
        Build the observation vector for this drone.

        Returns a flat numpy array containing:
          [0:3]   — position (x, y, z)
          [3:6]   — velocity (vx, vy, vz)
          [6]     — distance to nearest drone
          [7]     — distance to nearest obstacle surface
          [8]     — number of neighbours within communication range
          [9]     — altitude fraction (z / z_max)
        """
        # Nearest drone distance
        nearest_drone_dist = float("inf")
        neighbour_count = 0
        for other in all_drones:
            if other.id == self.id or not other.alive:
                continue
            dist = np.linalg.norm(self.position - other.position)
            if dist < nearest_drone_dist:
                nearest_drone_dist = dist
            if dist <= self.communication_range:
                neighbour_count += 1

        if nearest_drone_dist == float("inf"):
            nearest_drone_dist = 0.0

        # Nearest obstacle distance (axis-aligned bounding box)
        nearest_obs_dist = float("inf")
        for obs in obstacles:
            obs_pos = np.array(obs["position"])
            obs_size = np.array(obs["size"])
            # Closest point on AABB to drone center
            clamped = np.clip(
                self.position,
                obs_pos,
                obs_pos + obs_size,
            )
            dist = np.linalg.norm(self.position - clamped)
            if dist < nearest_obs_dist:
                nearest_obs_dist = dist

        if nearest_obs_dist == float("inf"):
            nearest_obs_dist = 0.0

        z_max = WORLD_CONFIG["world"]["z_max"]
        altitude_frac = self.position[2] / z_max if z_max > 0 else 0.0

        return np.array([
            *self.position,                # 0-2
            *self.velocity,                # 3-5
            nearest_drone_dist,            # 6
            nearest_obs_dist,              # 7
            float(neighbour_count),        # 8
            altitude_frac,                 # 9
        ], dtype=np.float32)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def reset(self, position: np.ndarray):
        """Reset drone to a new position with zero velocity."""
        self.position = position.copy()
        self.velocity = np.zeros(3)
        self.alive = True
        self.collisions = 0
        self.cells_explored = 0

    def to_dict(self) -> dict:
        """Serialise state for WebSocket streaming."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "alive": self.alive,
            "collisions": self.collisions,
        }

    def __repr__(self):
        return (
            f"Drone(id={self.id}, pos={self.position.round(2)}, "
            f"vel={self.velocity.round(2)}, alive={self.alive})"
        )
