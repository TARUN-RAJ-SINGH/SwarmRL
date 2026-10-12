"""
SwarmRL -- Day 11: Real-Time Simulation Engine & Policy Manager

Manages the SwarmEnv simulation lifecycle, policy inference (RLlib MAPPO checkpoint,
intelligent heuristic swarm controller, or random baseline), and packages
high-frequency telemetry frames for WebSocket streaming.
"""

import os
import sys
import glob
import asyncio
import numpy as np
import torch
from typing import Optional, Callable

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.swarm_env import SwarmEnv
from env.world_config import WORLD_CONFIG, WORLD_SIZE
from env.spaces import (
    LIDAR_HORIZONTAL_DIRS,
    LIDAR_VERTICAL_DIRS,
    SENSOR_RANGE,
    COMM_RANGE,
    cast_lidar_rays,
)

ALL_LIDAR_DIRS = np.concatenate([LIDAR_HORIZONTAL_DIRS, LIDAR_VERTICAL_DIRS], axis=0)


class PolicyManager:
    """
    Manages policy selection and inference for drone swarm control.
    Supports:
      - 'trained'  : RLlib PyTorch RLModule loaded from checkpoint
      - 'heuristic': Artificial Potential Field (APF) exploration & collision avoidance
      - 'random'   : Uniform random exploration baseline
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.mode = "heuristic"  # Default mode
        self.trained_module = None
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint(checkpoint_path)

    def _find_default_checkpoint(self) -> Optional[str]:
        """Find the latest trained checkpoint if available."""
        checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
        if not os.path.isdir(checkpoints_dir):
            return None

        # Look for learner rl_module shared_policy checkpoints
        patterns = [
            os.path.join(checkpoints_dir, "*", "learner_group", "learner", "rl_module", "shared_policy"),
            os.path.join(checkpoints_dir, "*_iter_*", "learner_group", "learner", "rl_module", "shared_policy"),
        ]
        candidates = []
        for p in patterns:
            candidates.extend(glob.glob(p))

        if candidates:
            # Sort by modification time descending
            candidates.sort(key=os.path.getmtime, reverse=True)
            return candidates[0]
        return None

    def _load_checkpoint(self, path: Optional[str] = None):
        """Attempts to load a trained PyTorch RLModule."""
        target_path = path or self._find_default_checkpoint()
        if not target_path or not os.path.isdir(target_path):
            print(f"[PolicyManager] No trained checkpoint found. Operating in heuristic mode.")
            self.mode = "heuristic"
            return

        try:
            from ray.rllib.core.rl_module.rl_module import RLModule
            print(f"[PolicyManager] Loading RLModule checkpoint from: {target_path}")
            self.trained_module = RLModule.from_checkpoint(target_path)
            self.trained_module.eval()
            self.checkpoint_path = target_path
            self.mode = "trained"
            print("[PolicyManager] Successfully loaded trained RL policy!")
        except Exception as e:
            print(f"[PolicyManager] Warning: Failed to load checkpoint ({e}). Falling back to heuristic.")
            self.trained_module = None
            self.mode = "heuristic"

    def set_mode(self, mode: str) -> str:
        """Switch policy mode between 'trained', 'heuristic', 'random'."""
        mode = mode.lower()
        if mode == "trained":
            if self.trained_module is None:
                self._load_checkpoint(self.checkpoint_path)
            if self.trained_module is not None:
                self.mode = "trained"
            else:
                self.mode = "heuristic"
        elif mode in ("heuristic", "random"):
            self.mode = mode
        else:
            raise ValueError(f"Unknown policy mode: {mode}")
        return self.mode

    def compute_actions(self, env: SwarmEnv, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Compute action dictionary for all active drones given observations.
        Returns dict mapping agent_name ('drone_0', etc.) -> np.ndarray of shape (3,).
        """
        agents = [a for a in env.agents if a in observations]
        if not agents:
            return {}

        if self.mode == "trained" and self.trained_module is not None:
            try:
                obs_batch = np.array([observations[a] for a in agents], dtype=np.float32)
                tensor_obs = torch.tensor(obs_batch, dtype=torch.float32)
                with torch.no_grad():
                    out = self.trained_module.forward_inference({"obs": tensor_obs})
                # action_dist_inputs has shape (batch_size, 6): [mean_0..2, log_std_0..2]
                raw_actions = out["action_dist_inputs"][:, :3].cpu().numpy()
                clipped_actions = np.clip(raw_actions, -1.0, 1.0)
                return {agent: clipped_actions[i] for i, agent in enumerate(agents)}
            except Exception as e:
                print(f"[PolicyManager] Inference error ({e}), using heuristic fallback.")
                return self._compute_heuristic_actions(env)

        elif self.mode == "heuristic":
            return self._compute_heuristic_actions(env)

        else:  # "random"
            return {agent: env.action_space(agent).sample() for agent in agents}

    def _compute_heuristic_actions(self, env: SwarmEnv) -> dict[str, np.ndarray]:
        """
        Artificial Potential Field (APF) policy:
          1. Inter-drone repulsion
          2. Obstacle & wall repulsion via LiDAR
          3. Radial exploration bias based on drone ID
        """
        actions = {}
        drones = env.drones
        world = WORLD_CONFIG["world"]

        for drone in drones:
            if not drone.alive:
                continue

            agent_name = f"drone_{drone.id}"
            pos = drone.position
            force = np.zeros(3, dtype=np.float32)

            # 1. Drone-Drone repulsion
            for other in drones:
                if other.id == drone.id or not other.alive:
                    continue
                diff = pos - other.position
                dist = np.linalg.norm(diff)
                if 0.01 < dist < COMM_RANGE:
                    repel_strength = (COMM_RANGE - dist) / COMM_RANGE
                    force += (diff / dist) * (repel_strength * 2.0)

            # 2. Obstacle & boundary repulsion via LiDAR
            lidar = cast_lidar_rays(pos.astype(np.float32), WORLD_CONFIG["obstacles"])
            for ray_idx, dist_norm in enumerate(lidar):
                if dist_norm < 0.4:  # Approaching obstacle or wall
                    ray_dir = ALL_LIDAR_DIRS[ray_idx]
                    push_mag = (0.4 - dist_norm) / 0.4 * 3.0
                    force -= ray_dir * push_mag

            # 3. World boundary soft margins
            margin = 10.0
            if pos[0] < world["x_min"] + margin:
                force[0] += 1.5
            elif pos[0] > world["x_max"] - margin:
                force[0] -= 1.5

            if pos[1] < world["y_min"] + margin:
                force[1] += 1.5
            elif pos[1] > world["y_max"] - margin:
                force[1] -= 1.5

            if pos[2] < world["z_min"] + 5.0:
                force[2] += 1.5
            elif pos[2] > world["z_max"] - 8.0:
                force[2] -= 1.5

            # 4. Outward exploration bias (different angle per drone)
            angle = (drone.id / max(len(drones), 1)) * 2 * np.pi
            explore_bias = np.array([np.cos(angle), np.sin(angle), 0.1], dtype=np.float32)
            force += explore_bias * 0.8

            # Normalize action to [-1, 1]
            mag = np.linalg.norm(force)
            if mag > 1.0:
                force = force / mag

            actions[agent_name] = force

        return actions


class SimulationEngine:
    """
    Core simulation runner. Wraps SwarmEnv, runs the physics loop at target FPS,
    gathers LiDAR hits & telemetry, and coordinates with WebSocket broadcasters.
    """

    def __init__(self, fps: int = 20, checkpoint_path: Optional[str] = None):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1)
        self.env = SwarmEnv()
        self.policy_manager = PolicyManager(checkpoint_path)

        self.status = "paused"  # "running", "paused", "completed"
        self.current_step = 0
        self.max_steps = WORLD_CONFIG["episode"]["max_steps"]
        self.latest_obs: dict[str, np.ndarray] = {}
        self.latest_infos: dict[str, dict] = {}
        self.latest_frame: dict = {}
        self.total_collisions = 0

        self._loop_task: Optional[asyncio.Task] = None
        self._broadcast_callbacks: list[Callable[[dict], None]] = []

        # Initialize environment state
        self.reset()

    def register_broadcast_callback(self, cb: Callable[[dict], None]):
        """Register a callback invoked whenever a new frame is generated."""
        self._broadcast_callbacks.append(cb)

    def remove_broadcast_callback(self, cb: Callable[[dict], None]):
        if cb in self._broadcast_callbacks:
            self._broadcast_callbacks.remove(cb)

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset environment and generate initial frame."""
        self.latest_obs, self.latest_infos = self.env.reset(seed=seed)
        self.current_step = 0
        self.total_collisions = 0
        self.status = "paused"

        empty_actions = {f"drone_{d.id}": np.zeros(3) for d in self.env.drones}
        self.latest_frame = self._build_frame(
            actions=empty_actions,
            new_cells=0,
            dd_cols=[],
            do_cols=[],
            is_terminated=False,
            is_truncated=False,
        )
        return self.latest_frame

    def step(self) -> dict:
        """Advance the simulation by one step."""
        if not self.env.agents:
            self.reset()

        actions = self.policy_manager.compute_actions(self.env, self.latest_obs)
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self.latest_obs = next_obs
        self.latest_infos = infos
        self.current_step = self.env.current_step

        # Gather collision info from processor
        step_result = getattr(self.env, "processor", None)
        dd_cols = []
        do_cols = []
        new_cells = 0
        if self.env.drones:
            first_info = infos.get(f"drone_{self.env.drones[0].id}", {})
            new_cells = first_info.get("cells_explored", 0)

        # Check termination or truncation
        is_terminated = any(terminations.values()) if terminations else False
        is_truncated = any(truncations.values()) if truncations else False

        # Count collisions
        for d in self.env.drones:
            self.total_collisions += infos.get(f"drone_{d.id}", {}).get("collisions", 0)

        if is_terminated or is_truncated:
            self.status = "completed"

        self.latest_frame = self._build_frame(
            actions=actions,
            new_cells=new_cells,
            dd_cols=dd_cols,
            do_cols=do_cols,
            is_terminated=is_terminated,
            is_truncated=is_truncated,
        )
        return self.latest_frame

    def _build_frame(
        self,
        actions: dict,
        new_cells: int,
        dd_cols: list,
        do_cols: list,
        is_terminated: bool,
        is_truncated: bool,
    ) -> dict:
        """Package comprehensive frame dictionary for WebSocket streaming."""
        drones_data = []
        coverage = float(np.sum(self.env.coverage_grid)) / max(self.env.coverage_grid.size, 1)

        for drone in self.env.drones:
            agent_name = f"drone_{drone.id}"
            action = actions.get(agent_name, np.zeros(3))
            if isinstance(action, np.ndarray):
                action = action.tolist()

            pos = drone.position
            # Compute LiDAR readings and 3D hit point coordinates for rendering
            lidar_dists = cast_lidar_rays(pos.astype(np.float32), WORLD_CONFIG["obstacles"])
            lidar_hits = []
            for i, r_norm in enumerate(lidar_dists):
                ray_dir = ALL_LIDAR_DIRS[i]
                hit_coord = pos + ray_dir * (r_norm * SENSOR_RANGE)
                lidar_hits.append([round(float(c), 2) for c in hit_coord])

            drones_data.append({
                "id": drone.id,
                "position": [round(float(p), 2) for p in pos],
                "velocity": [round(float(v), 2) for v in drone.velocity],
                "speed": round(float(np.linalg.norm(drone.velocity)), 2),
                "alive": bool(drone.alive),
                "collisions": int(drone.collisions),
                "cells_explored": int(drone.cells_explored),
                "lidar_ranges": [round(float(r), 3) for r in lidar_dists],
                "lidar_hits": lidar_hits,
                "action": [round(float(a), 2) for a in action],
            })

        return {
            "type": "state_update",
            "step": self.current_step,
            "max_steps": self.max_steps,
            "coverage": round(coverage, 4),
            "coverage_pct": round(coverage * 100, 2),
            "new_cells": new_cells,
            "total_collisions": self.total_collisions,
            "status": self.status,
            "policy_mode": self.policy_manager.mode,
            "fps": self.fps,
            "drones": drones_data,
            "events": {
                "drone_drone_collisions": dd_cols,
                "drone_obstacle_collisions": do_cols,
            },
            "is_terminated": is_terminated,
            "is_truncated": is_truncated,
        }

    async def start(self):
        """Start or resume the continuous simulation loop."""
        if self.status == "completed":
            self.reset()

        self.status = "running"
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._simulation_loop())

    def pause(self):
        """Pause the continuous simulation loop."""
        self.status = "paused"

    def set_fps(self, fps: int):
        """Adjust simulation tick rate."""
        self.fps = max(1, min(fps, 60))
        self.dt = 1.0 / self.fps

    async def _simulation_loop(self):
        """Async loop running simulation steps at target FPS."""
        try:
            while self.status == "running":
                frame = self.step()

                # Dispatch frame to all broadcast callbacks
                for cb in self._broadcast_callbacks:
                    try:
                        res = cb(frame)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        print(f"[SimulationEngine] Broadcast callback error: {e}")

                if frame["is_terminated"] or frame["is_truncated"]:
                    self.status = "completed"
                    break

                await asyncio.sleep(self.dt)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[SimulationEngine] Loop exception: {e}")
            self.status = "paused"
