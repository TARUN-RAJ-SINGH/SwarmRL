"""
SwarmRL -- Day 15: CommSwarmEnv

A thin wrapper around SwarmEnv that augments every drone's observation
with the 16-dim aggregated communication context produced by gnn_comm.py.

Observation space: 42 dimensions (26 base + 16 comm context)
Action space:      3 dimensions (unchanged)

Optional features:
  - Packet-loss simulation (loss_prob parameter)
  - Bandwidth cap (max_msgs parameter)
  - Formation-flight bonus reward for tight cooperative dispersion
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.swarm_env import SwarmEnv
from env.gnn_comm import (
    build_comm_observation,
    ENHANCED_OBS_DIM,
    MSG_DIM,
)
from env.spaces import get_action_space, OBS_DIM
from env.world_config import WORLD_CONFIG


def get_enhanced_observation_space() -> spaces.Box:
    """
    Returns Gymnasium Box for the 42-dim enhanced observation.
    Base 26 dims use same bounds as Day 3; comm context 16 dims in [-1,1].
    """
    # Base observation bounds
    base_low  = np.array([
        0.0, 0.0, 0.0,          # pos xyz         [0-2]
        -1.0, -1.0, -1.0,       # vel xyz          [3-5]
        0.0,                    # speed_frac        [6]
        0.0,                    # alt_frac          [7]
        0.0,                    # nearest_drone     [8]
        0.0,                    # nearest_obstacle  [9]
        0.0,                    # neighbour_count   [10]
        *([0.0] * 10),          # LiDAR 10 rays    [11-20]
        0.0,                    # local_cov        [21]
        0.0,                    # global_cov       [22]
        0.0,                    # wall_prox        [23]
        0.0,                    # steps_frac       [24]
        0.0,                    # heading          [25]
    ], dtype=np.float32)

    base_high = np.ones(OBS_DIM, dtype=np.float32)
    base_high[3:6] = 1.0  # velocity can be ±1
    base_low[3:6]  = -1.0

    # Comm context bounds: relative positions [-1,1], rest in [-1,1]
    ctx_low  = -np.ones(MSG_DIM, dtype=np.float32)
    ctx_high =  np.ones(MSG_DIM, dtype=np.float32)

    low  = np.concatenate([base_low,  ctx_low])
    high = np.concatenate([base_high, ctx_high])

    assert low.shape == (ENHANCED_OBS_DIM,)
    return spaces.Box(low=low, high=high, shape=(ENHANCED_OBS_DIM,), dtype=np.float32)


class CommSwarmEnv(ParallelEnv):
    """
    PettingZoo Parallel environment with GNN inter-agent communication.

    Wraps SwarmEnv: every reset/step call runs the inner env, then
    enriches each drone's observation with a spatial-attention-aggregated
    message from all in-range neighbours.

    Additional reward shaping:
      - Formation bonus: +0.05 if mean inter-drone separation is in the
        cooperative sweet spot (sensor_range/2 to comm_range/2).
        Encourages spreading without isolation.
    """

    metadata = {"render_modes": ["human"], "name": "comm_swarmrl_v0"}

    def __init__(
        self,
        render_mode:  str   | None = None,
        num_drones:   int   | None = None,
        loss_prob:    float        = 0.0,
        max_msgs:     int          = 9,
        formation_bonus: bool      = True,
    ):
        """
        Args:
            render_mode:     Passed through to inner SwarmEnv.
            num_drones:      Fleet size override.
            loss_prob:       Packet-loss probability per comm link [0,1].
            max_msgs:        Max neighbours aggregated per step (bandwidth cap).
            formation_bonus: Enable cooperative dispersion reward shaping.
        """
        super().__init__()

        self._inner = SwarmEnv(render_mode=render_mode, num_drones=num_drones)
        self.loss_prob       = loss_prob
        self.max_msgs        = max_msgs
        self.formation_bonus = formation_bonus

        self.cfg          = WORLD_CONFIG
        self.comm_range   = self.cfg["drones"]["communication_range"]  # 20m
        self.sensor_range = self.cfg["drones"]["sensor_range"]         # 10m

        # Cooperative dispersion sweet-spot bounds
        self._disp_lo = self.sensor_range / 2.0    # 5m  — min healthy sep
        self._disp_hi = self.comm_range   / 2.0    # 10m — max comm radius

        # PettingZoo API attributes
        self.possible_agents = self._inner.possible_agents[:]
        self.agents          = self.possible_agents[:]

        _enh_obs_space = get_enhanced_observation_space()
        self.observation_spaces = {a: _enh_obs_space for a in self.possible_agents}
        self.action_spaces      = {a: get_action_space() for a in self.possible_agents}

    # ──────────────────────────────────────────────────────────
    # PettingZoo API
    # ──────────────────────────────────────────────────────────

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        base_obs, infos = self._inner.reset(seed=seed, options=options)
        self.agents = self._inner.agents[:]
        enhanced_obs = self._build_comm_obs(base_obs)
        return enhanced_obs, infos

    def step(self, actions):
        base_obs, rewards, terminations, truncations, infos = self._inner.step(actions)
        self.agents = self._inner.agents[:]

        enhanced_obs = self._build_comm_obs(base_obs)

        # Formation bonus
        if self.formation_bonus and len(self._inner.drones) > 1:
            bonus = self._compute_formation_bonus()
            for agent in rewards:
                rewards[agent] += bonus

        return enhanced_obs, rewards, terminations, truncations, infos

    # ──────────────────────────────────────────────────────────
    # Communication Observation Builder
    # ──────────────────────────────────────────────────────────

    def _build_comm_obs(self, base_obs: dict) -> dict:
        """
        Enrich each drone's 26-dim obs with 16-dim aggregated comm context.
        Returns dict of agent_name → 42-dim enhanced observation.
        """
        drones = self._inner.drones

        # Map drone_id → base obs (numpy)
        id_to_obs = {}
        for agent_name, obs in base_obs.items():
            did = int(agent_name.split("_")[1])
            id_to_obs[did] = obs

        # Map drone_id → world position
        id_to_pos = {d.id: d.position.copy() for d in drones}

        enhanced = {}
        for agent_name, obs in base_obs.items():
            did = int(agent_name.split("_")[1])
            enh = build_comm_observation(
                drone_id        = did,
                base_obs        = obs,
                all_obs         = id_to_obs,
                drone_positions = id_to_pos,
                comm_range      = self.comm_range,
                sensor_range    = self.sensor_range,
                loss_prob       = self.loss_prob,
                max_msgs        = self.max_msgs,
            )
            enhanced[agent_name] = enh

        return enhanced

    def _compute_formation_bonus(self) -> float:
        """
        Returns a small reward bonus if the swarm mean separation is in
        the cooperative dispersion sweet-spot [_disp_lo, _disp_hi].
        """
        drones = self._inner.drones
        if len(drones) < 2:
            return 0.0

        positions = np.array([d.position for d in drones if d.alive])
        if len(positions) < 2:
            return 0.0

        n = len(positions)
        total_sep = 0.0
        count     = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sep += float(np.linalg.norm(positions[i] - positions[j]))
                count     += 1

        mean_sep = total_sep / max(count, 1)

        if self._disp_lo <= mean_sep <= self._disp_hi:
            return 0.05   # formation bonus
        return 0.0

    # ──────────────────────────────────────────────────────────
    # Convenience / Evaluation Accessors
    # ──────────────────────────────────────────────────────────

    @property
    def inner_env(self) -> SwarmEnv:
        """Direct access to the underlying SwarmEnv."""
        return self._inner

    @property
    def current_step(self) -> int:
        return self._inner.current_step

    @property
    def drones(self):
        return self._inner.drones

    @property
    def coverage_grid(self):
        return self._inner.coverage_grid

    def get_comm_stats(self) -> dict:
        """
        Return current communication topology statistics:
          - mean_neighbours: average in-range neighbours per drone
          - isolated_drones: number of drones with 0 in-range neighbours
          - mean_separation: mean pairwise distance (m)
        """
        drones = self._inner.drones
        alive  = [d for d in drones if d.alive]
        if len(alive) < 2:
            return {"mean_neighbours": 0.0, "isolated_drones": len(alive), "mean_separation": 0.0}

        positions = np.array([d.position for d in alive])
        n = len(alive)

        neighbour_counts = []
        separations = []
        for i, d in enumerate(alive):
            count = 0
            for j, o in enumerate(alive):
                if i == j:
                    continue
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                separations.append(dist)
                if dist <= self.comm_range:
                    count += 1
            neighbour_counts.append(count)

        return {
            "mean_neighbours":  float(np.mean(neighbour_counts)),
            "isolated_drones":  int(np.sum(np.array(neighbour_counts) == 0)),
            "mean_separation":  float(np.mean(separations)) if separations else 0.0,
        }

    def get_global_state(self) -> dict:
        """
        Return the centralized global state for CTDE training,
        enriched with communication topology statistics.
        """
        state = self._inner.get_global_state()
        state["comm_stats"] = self.get_comm_stats()
        return state
