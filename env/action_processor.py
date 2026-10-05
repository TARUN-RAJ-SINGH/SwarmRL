"""
SwarmRL -- Day 4: Action Processor

Bridges the normalised action space [-1, 1] from the RL policy to the
drone physics engine. Handles:

  1. Scaling normalised actions to real accelerations
  2. Optional action smoothing (exponential moving average) for stability
  3. Action masking for dead drones
  4. Full step pipeline: action -> physics -> collision -> observation
"""

import numpy as np
from env.world_config import WORLD_CONFIG
from env.drone import Drone
from env.collisions import (
    check_drone_drone_collisions,
    check_drone_obstacle_collisions,
    resolve_drone_drone_collision,
    resolve_drone_obstacle_collision,
)
from env.spaces import (
    build_observation,
    MAX_ACCEL,
    OBS_DIM,
    GRID_CELLS_PER_AXIS,
    WORLD_ORIGIN,
)


class ActionProcessor:
    """
    Processes normalised RL actions and steps the drone simulation forward.

    Provides the complete action -> physics -> collision -> observation loop
    that the PettingZoo environment will use in its step() function.
    """

    def __init__(self, smoothing: float = 0.0):
        """
        Args:
            smoothing: Exponential moving average factor in [0, 1).
                       0.0 = no smoothing (raw actions applied directly)
                       0.5 = equal blend of previous and current action
                       Higher values = smoother but more sluggish response
        """
        self.smoothing = smoothing
        self._prev_actions: dict[int, np.ndarray] = {}  # drone_id -> last action

        self.dt = WORLD_CONFIG["episode"]["dt"]
        self.obstacles = WORLD_CONFIG["obstacles"]
        self.max_steps = WORLD_CONFIG["episode"]["max_steps"]

    # ──────────────────────────────────────────────
    # Action Processing
    # ──────────────────────────────────────────────

    def scale_action(self, normalised_action: np.ndarray) -> np.ndarray:
        """
        Convert a normalised action in [-1, 1] to real acceleration in m/s^2.

        Args:
            normalised_action: np.array of shape (3,) with values in [-1, 1]

        Returns:
            np.array of shape (3,) — acceleration in [-max_accel, max_accel]
        """
        clipped = np.clip(normalised_action, -1.0, 1.0)
        return clipped * MAX_ACCEL

    def smooth_action(self, drone_id: int, action: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average smoothing to reduce jitter.

        Args:
            drone_id: ID of the drone
            action: Current raw scaled action

        Returns:
            Smoothed action
        """
        if self.smoothing <= 0.0 or drone_id not in self._prev_actions:
            self._prev_actions[drone_id] = action.copy()
            return action

        alpha = self.smoothing
        smoothed = alpha * self._prev_actions[drone_id] + (1 - alpha) * action
        self._prev_actions[drone_id] = smoothed.copy()
        return smoothed

    def process_action(self, drone_id: int, normalised_action: np.ndarray) -> np.ndarray:
        """
        Full action pipeline: clip -> scale -> smooth.

        Args:
            drone_id: ID of the drone
            normalised_action: np.array of shape (3,) in [-1, 1]

        Returns:
            Final acceleration vector in m/s^2
        """
        scaled = self.scale_action(normalised_action)
        smoothed = self.smooth_action(drone_id, scaled)
        return smoothed

    # ──────────────────────────────────────────────
    # Simulation Step
    # ──────────────────────────────────────────────

    def step_drone(self, drone: Drone, normalised_action: np.ndarray):
        """
        Apply a single action to a single drone: process action, update
        physics, and clamp to world bounds.

        Args:
            drone: The Drone instance
            normalised_action: np.array of shape (3,) in [-1, 1]

        Returns:
            The real acceleration that was applied (for logging)
        """
        if not drone.alive:
            return np.zeros(3)

        real_accel = self.process_action(drone.id, normalised_action)
        drone.apply_action(real_accel, self.dt)
        drone.clamp_to_world()
        return real_accel

    def step_all(
        self,
        drones: list[Drone],
        actions: dict[int, np.ndarray],
        coverage_grid: np.ndarray,
        current_step: int,
    ) -> dict:
        """
        Step ALL drones forward by one timestep. This is the main loop
        the PettingZoo environment will call.

        Pipeline:
          1. Apply actions to each drone (scale + physics + clamp)
          2. Detect and resolve drone-drone collisions
          3. Detect and resolve drone-obstacle collisions
          4. Update coverage grid
          5. Build observations for all drones

        Args:
            drones:        List of all Drone instances
            actions:       Dict mapping drone_id -> normalised action (3,)
            coverage_grid: 3D boolean array of explored voxels
            current_step:  Current step number in the episode

        Returns:
            Dict with keys:
              "observations"   — dict[drone_id, np.ndarray]
              "collisions"     — {"drone_drone": [...], "drone_obstacle": [...]}
              "coverage"       — float (global coverage fraction)
              "new_cells"      — int (newly explored cells this step)
              "applied_accels" — dict[drone_id, np.ndarray]
        """
        applied_accels = {}

        # 1. Apply actions
        for drone in drones:
            action = actions.get(drone.id, np.zeros(3))
            accel = self.step_drone(drone, action)
            applied_accels[drone.id] = accel

        # 2. Drone-drone collisions
        dd_collisions = check_drone_drone_collisions(drones)
        for id_a, id_b in dd_collisions:
            drone_a = next(d for d in drones if d.id == id_a)
            drone_b = next(d for d in drones if d.id == id_b)
            resolve_drone_drone_collision(drone_a, drone_b)

        # 3. Drone-obstacle collisions
        do_collisions = check_drone_obstacle_collisions(drones, self.obstacles)
        for drone_id, obs_id in do_collisions:
            drone = next(d for d in drones if d.id == drone_id)
            obs = next(o for o in self.obstacles if o["id"] == obs_id)
            resolve_drone_obstacle_collision(drone, obs)

        # 4. Update coverage grid
        new_cells = self._update_coverage(drones, coverage_grid)
        total_cells = max(coverage_grid.size, 1)
        global_coverage = float(np.sum(coverage_grid)) / total_cells

        # 5. Build observations
        steps_remaining = self.max_steps - current_step
        observations = {}
        for drone in drones:
            observations[drone.id] = build_observation(
                drone=drone,
                all_drones=drones,
                obstacles=self.obstacles,
                coverage_grid=coverage_grid,
                global_coverage_frac=global_coverage,
                steps_remaining=steps_remaining,
                max_steps=self.max_steps,
            )

        return {
            "observations": observations,
            "collisions": {
                "drone_drone": dd_collisions,
                "drone_obstacle": do_collisions,
            },
            "coverage": global_coverage,
            "new_cells": new_cells,
            "applied_accels": applied_accels,
        }

    # ──────────────────────────────────────────────
    # Coverage Update
    # ──────────────────────────────────────────────

    def _update_coverage(self, drones: list[Drone], coverage_grid: np.ndarray) -> int:
        """
        Mark voxels around each drone as explored based on sensor range.
        Returns the number of newly explored cells this step.
        """
        res = WORLD_CONFIG["coverage"]["grid_resolution"]
        sensor_range = WORLD_CONFIG["drones"]["sensor_range"]
        radius_cells = int(np.ceil(sensor_range / res))
        new_cells = 0

        for drone in drones:
            if not drone.alive:
                continue

            # Convert position to grid indices
            grid_pos = ((drone.position - WORLD_ORIGIN) / res).astype(int)

            # Compute bounds within grid
            lo = np.maximum(grid_pos - radius_cells, 0)
            hi = np.minimum(grid_pos + radius_cells + 1, GRID_CELLS_PER_AXIS)

            if np.any(hi <= lo):
                continue

            # Count new cells before marking
            local_slice = coverage_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            unexplored = np.sum(~local_slice)

            # Mark as explored
            coverage_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True

            new_cells += int(unexplored)
            drone.cells_explored += int(unexplored)

        return new_cells

    def reset(self):
        """Clear action history (call on episode reset)."""
        self._prev_actions.clear()
