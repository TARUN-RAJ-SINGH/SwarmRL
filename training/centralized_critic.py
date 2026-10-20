"""
SwarmRL -- Day 16: Centralized Critic Architecture for CTDE

Implements the Centralized Training with Decentralized Execution (CTDE)
value function architecture for multi-agent drone swarms:

1. GlobalStateBuilder:
   Extracts and normalizes the full joint environment state:
     - Agent entity matrix (N, 12): pos, vel, speed, alt, nearest drone,
       nearest obstacle, cells explored, alive status
     - Global env context (11): global coverage, Zone A coverage, Zone B coverage,
       total collisions, step fraction, swarm centroid, dispersion radius,
       alive ratio, mean fleet speed
     - Supports variable fleet size (N in [10, 25, 50]) with optional fixed padding

2. EntityAttentionCritic (PyTorch Module):
   Permutation-invariant centralized critic using multi-head self-attention
   and DeepSets pooling (mean + max). Invariant to agent ordering and dynamically
   scales across fleet sizes without retraining network parameters.
   Predicts joint swarm value baseline V(s) and optional agent-specific baselines V_i(s).

3. MLPCentralizedCritic (PyTorch Module):
   Standard feed-forward MLP baseline critic for flattened global state vectors.

4. ValueNorm (PopArt):
   Adaptive value target normalization with PopArt (Preserving Outputs Precisely
   while Adaptively Rescaling Targets) exact weight/bias adjustment to stabilize
   learning under non-stationary multi-agent reward regimes.

5. MultiAgentGAE:
   Multi-Agent Generalized Advantage Estimation (lambda-returns) with advantage
   normalization for cooperative credit assignment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.world_config import (
    WORLD_CONFIG,
    WORLD_SIZE,
    GRID_CELLS_PER_AXIS,
    TOTAL_GRID_CELLS,
)
from env.spaces import (
    WORLD_ORIGIN,
    WORLD_EXTENT,
    MAX_SPEED,
    MAX_ACCEL,
    SENSOR_RANGE,
    COMM_RANGE,
)
from env.drone import Drone


# ==============================================================================
# Architecture Constants & Dimensions
# ==============================================================================

AGENT_FEATURE_DIM = 12
"""
Per-drone entity features (12 dimensions):
  [0:3]   Position (x, y, z) normalized to [0, 1]
  [3:6]   Velocity (vx, vy, vz) normalized to [-1, 1]
  [6]     Speed fraction (|v| / max_speed) in [0, 1]
  [7]     Altitude fraction (z / z_max) in [0, 1]
  [8]     Nearest drone distance normalized by world diagonal in [0, 1]
  [9]     Nearest obstacle surface distance normalized by sensor_range in [0, 1]
  [10]    Cells explored fraction (cells / 1000) clipped in [0, 1]
  [11]    Alive flag (1.0 = alive, 0.0 = inactive)
"""

ENV_FEATURE_DIM = 11
"""
Global environment context (11 dimensions):
  [0]     Global coverage fraction in [0, 1]
  [1]     Zone A priority sector coverage fraction in [0, 1]
  [2]     Zone B secondary sector coverage fraction in [0, 1]
  [3]     Total fleet collisions normalized in [0, 1]
  [4]     Step progress fraction (current_step / max_steps) in [0, 1]
  [5:8]   Swarm centroid (Cx, Cy, Cz) normalized to [0, 1]
  [8]     Swarm dispersion radius normalized by world diagonal in [0, 1]
  [9]     Fleet alive ratio (active / total) in [0, 1]
  [10]    Fleet mean speed fraction in [0, 1]
"""

MAX_DRONES = 50
MAX_FLAT_STATE_DIM = MAX_DRONES * AGENT_FEATURE_DIM + ENV_FEATURE_DIM  # 611

# World diagonal: sqrt(100^2 + 100^2 + 50^2) = 150.0 m
WORLD_DIAG = float(np.linalg.norm(WORLD_EXTENT))


# ==============================================================================
# 1. Global State Builder
# ==============================================================================

class GlobalStateBuilder:
    """
    Extracts, normalizes, and packages the global multi-agent state
    for Centralized Critic ingestion during CTDE training.
    """

    def __init__(self, max_drones: int = MAX_DRONES):
        self.max_drones = int(max_drones)
        self.world_origin = WORLD_ORIGIN.astype(np.float32)
        self.world_extent = WORLD_EXTENT.astype(np.float32)
        self.world_diag = WORLD_DIAG
        self.max_speed = float(MAX_SPEED)
        self.sensor_range = float(SENSOR_RANGE)

        # Obstacle AABBs for vectorized closest-point calculation
        obstacles = WORLD_CONFIG.get("obstacles", [])
        if obstacles:
            self.obs_mins = np.array([o["position"] for o in obstacles], dtype=np.float32)
            self.obs_maxs = np.array(
                [np.array(o["position"]) + np.array(o["size"]) for o in obstacles],
                dtype=np.float32,
            )
        else:
            self.obs_mins = np.zeros((0, 3), dtype=np.float32)
            self.obs_maxs = np.zeros((0, 3), dtype=np.float32)

        # Precompute 2D search zone masks for fast voxel coverage evaluation
        self._init_search_zone_masks()

    def _init_search_zone_masks(self):
        """Precomputes 2D projection masks for Zone A and Zone B."""
        res = float(WORLD_CONFIG["coverage"]["grid_resolution"])
        cells_x = int(GRID_CELLS_PER_AXIS[0])
        cells_y = int(GRID_CELLS_PER_AXIS[1])
        cells_z = int(GRID_CELLS_PER_AXIS[2])

        x_coords = (np.arange(cells_x, dtype=np.float32) + 0.5) * res + self.world_origin[0]
        y_coords = (np.arange(cells_y, dtype=np.float32) + 0.5) * res + self.world_origin[1]
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

        search_zones = WORLD_CONFIG.get("search_zones", [])
        # Zone A
        if len(search_zones) > 0:
            za = search_zones[0]
            za_center = np.array(za["center"][:2], dtype=np.float32)
            za_radius = float(za["radius"])
            self.zone_a_mask = ((xx - za_center[0]) ** 2 + (yy - za_center[1]) ** 2) <= (za_radius ** 2)
            self.zone_a_cell_count = max(int(np.count_nonzero(self.zone_a_mask) * cells_z), 1)
        else:
            self.zone_a_mask = np.zeros((cells_x, cells_y), dtype=bool)
            self.zone_a_cell_count = 1

        # Zone B
        if len(search_zones) > 1:
            zb = search_zones[1]
            zb_center = np.array(zb["center"][:2], dtype=np.float32)
            zb_radius = float(zb["radius"])
            self.zone_b_mask = ((xx - zb_center[0]) ** 2 + (yy - zb_center[1]) ** 2) <= (zb_radius ** 2)
            self.zone_b_cell_count = max(int(np.count_nonzero(self.zone_b_mask) * cells_z), 1)
        else:
            self.zone_b_mask = np.zeros((cells_x, cells_y), dtype=bool)
            self.zone_b_cell_count = 1

    def compute_zone_coverages(self, coverage_grid: np.ndarray) -> Tuple[float, float]:
        """
        Fast 2D projected voxel summation for Zone A and Zone B coverage fractions.
        """
        if coverage_grid is None or coverage_grid.size == 0:
            return 0.0, 0.0

        grid_2d = np.sum(coverage_grid, axis=2)  # shape (cells_x, cells_y)
        exp_a = float(np.sum(grid_2d[self.zone_a_mask]))
        exp_b = float(np.sum(grid_2d[self.zone_b_mask]))

        cov_a = float(np.clip(exp_a / self.zone_a_cell_count, 0.0, 1.0))
        cov_b = float(np.clip(exp_b / self.zone_b_cell_count, 0.0, 1.0))
        return cov_a, cov_b

    def build_from_env(self, env: Any) -> Dict[str, Any]:
        """
        Extract global state directly from a SwarmEnv or CommSwarmEnv instance.
        """
        # Unwrap CommSwarmEnv if necessary
        inner = getattr(env, "_inner", env)
        drones = inner.drones
        coverage_grid = inner.coverage_grid
        current_step = inner.current_step
        max_steps = inner.max_steps
        return self.build(
            drones=drones,
            coverage_grid=coverage_grid,
            current_step=current_step,
            max_steps=max_steps,
        )

    def build(
        self,
        drones: list[Drone],
        coverage_grid: np.ndarray,
        current_step: int = 0,
        max_steps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Builds normalized agent entity matrix, environment context vector,
        and flat/padded joint state representations.

        Args:
            drones: List of Drone objects
            coverage_grid: 3D boolean voxel array (50, 50, 25)
            current_step: Current simulation step index
            max_steps: Maximum episode steps

        Returns:
            Dictionary containing:
              - 'agent_states': np.ndarray (N, 12)
              - 'env_state': np.ndarray (11,)
              - 'flat_state': np.ndarray (N * 12 + 11,)
              - 'padded_state': np.ndarray (max_drones * 12 + 11,)
              - 'agent_mask': np.ndarray (max_drones,) boolean
              - 'num_drones': int
              - 'coverage': float
              - 'centroid': np.ndarray (3,)
              - 'dispersion': float
        """
        num_drones = len(drones)
        agent_states = np.zeros((num_drones, AGENT_FEATURE_DIM), dtype=np.float32)

        if num_drones > 0:
            positions = np.array([d.position for d in drones], dtype=np.float32)
            velocities = np.array([d.velocity for d in drones], dtype=np.float32)
            alive_flags = np.array([1.0 if d.alive else 0.0 for d in drones], dtype=np.float32)
            cells_exp = np.array([float(d.cells_explored) for d in drones], dtype=np.float32)

            # 1. Position normalized [0, 1]
            pos_norm = np.clip(
                (positions - self.world_origin[None, :]) / self.world_extent[None, :],
                0.0,
                1.0,
            )
            agent_states[:, 0:3] = pos_norm

            # 2. Velocity normalized [-1, 1]
            vel_norm = np.clip(velocities / self.max_speed, -1.0, 1.0)
            agent_states[:, 3:6] = vel_norm

            # 3. Speed fraction [0, 1]
            speeds = np.linalg.norm(velocities, axis=-1)
            agent_states[:, 6] = np.clip(speeds / self.max_speed, 0.0, 1.0)

            # 4. Altitude fraction [0, 1]
            agent_states[:, 7] = pos_norm[:, 2]

            # 5. Nearest drone distance normalized [0, 1]
            if num_drones > 1:
                diffs = positions[:, None, :] - positions[None, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_drone_dist = np.min(dists, axis=-1)
            else:
                min_drone_dist = np.array([self.world_diag], dtype=np.float32)
            agent_states[:, 8] = np.clip(min_drone_dist / self.world_diag, 0.0, 1.0)

            # 6. Nearest obstacle distance normalized [0, 1]
            if len(self.obs_mins) > 0:
                closest_pts = np.clip(
                    positions[:, None, :],
                    self.obs_mins[None, :, :],
                    self.obs_maxs[None, :, :],
                )
                obs_dists = np.linalg.norm(positions[:, None, :] - closest_pts, axis=-1)
                min_obs_dist = np.min(obs_dists, axis=-1)
            else:
                min_obs_dist = np.full(num_drones, self.sensor_range, dtype=np.float32)
            agent_states[:, 9] = np.clip(min_obs_dist / self.sensor_range, 0.0, 1.0)

            # 7. Cells explored fraction [0, 1]
            agent_states[:, 10] = np.clip(cells_exp / 1000.0, 0.0, 1.0)

            # 8. Alive flag [0, 1]
            agent_states[:, 11] = alive_flags

        # Global Environment Context (11 dims)
        env_state = np.zeros(ENV_FEATURE_DIM, dtype=np.float32)

        # Global coverage
        if coverage_grid is not None and coverage_grid.size > 0:
            global_cov = float(np.count_nonzero(coverage_grid) / TOTAL_GRID_CELLS)
        else:
            global_cov = 0.0
        env_state[0] = float(np.clip(global_cov, 0.0, 1.0))

        # Search zone coverages
        cov_a, cov_b = self.compute_zone_coverages(coverage_grid)
        env_state[1] = cov_a
        env_state[2] = cov_b

        # Fleet collisions
        total_cols = sum(d.collisions for d in drones) if drones else 0
        max_col_denom = max(num_drones * 5, 1)
        env_state[3] = float(np.clip(total_cols / max_col_denom, 0.0, 1.0))

        # Step fraction
        env_state[4] = float(np.clip(current_step / max(max_steps, 1), 0.0, 1.0))

        # Swarm Centroid & Dispersion
        if num_drones > 0:
            alive_drones = [d for d in drones if d.alive]
            if alive_drones:
                alive_pos = np.array([d.position for d in alive_drones], dtype=np.float32)
                alive_speeds = np.array(
                    [np.linalg.norm(d.velocity) / self.max_speed for d in alive_drones],
                    dtype=np.float32,
                )
            else:
                alive_pos = positions
                alive_speeds = np.zeros(num_drones, dtype=np.float32)

            centroid = np.mean(alive_pos, axis=0)
            norm_centroid = np.clip(
                (centroid - self.world_origin) / self.world_extent, 0.0, 1.0
            )
            env_state[5:8] = norm_centroid

            dispersion = float(np.sqrt(np.mean(np.sum((alive_pos - centroid) ** 2, axis=-1))))
            norm_dispersion = float(np.clip(dispersion / self.world_diag, 0.0, 1.0))
            env_state[8] = norm_dispersion

            alive_ratio = float(len(alive_drones) / max(num_drones, 1))
            env_state[9] = alive_ratio
            env_state[10] = float(np.clip(np.mean(alive_speeds), 0.0, 1.0))
        else:
            centroid = np.zeros(3, dtype=np.float32)
            dispersion = 0.0

        # Unpadded Flat State: (N * 12 + 11,)
        flat_state = np.concatenate([agent_states.flatten(), env_state], dtype=np.float32)

        # Fixed Padded State: (max_drones * 12 + 11,) for fixed-size buffer architectures
        padded_agents = np.zeros((self.max_drones, AGENT_FEATURE_DIM), dtype=np.float32)
        agent_mask = np.zeros(self.max_drones, dtype=bool)

        valid_count = min(num_drones, self.max_drones)
        if valid_count > 0:
            padded_agents[:valid_count] = agent_states[:valid_count]
            agent_mask[:valid_count] = True

        padded_state = np.concatenate([padded_agents.flatten(), env_state], dtype=np.float32)

        return {
            "agent_states": agent_states,
            "env_state": env_state,
            "flat_state": flat_state,
            "padded_state": padded_state,
            "agent_mask": agent_mask,
            "num_drones": num_drones,
            "coverage": global_cov,
            "zone_a_coverage": cov_a,
            "zone_b_coverage": cov_b,
            "centroid": centroid,
            "dispersion": dispersion,
        }

    @staticmethod
    def unpack_flat_state(
        flat_state: Union[np.ndarray, torch.Tensor],
        num_drones: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Decomposes a flattened global state vector into agent_states (N, 12)
        and env_state (11,). Supports both batched and unbatched inputs.
        """
        is_torch = isinstance(flat_state, torch.Tensor)
        last_dim = flat_state.shape[-1]
        raw_agent_features = last_dim - ENV_FEATURE_DIM

        if raw_agent_features % AGENT_FEATURE_DIM != 0:
            raise ValueError(
                f"Invalid flat_state dimension {last_dim}: remaining {raw_agent_features} "
                f"features not divisible by AGENT_FEATURE_DIM ({AGENT_FEATURE_DIM})"
            )

        n_agents = raw_agent_features // AGENT_FEATURE_DIM if num_drones is None else num_drones

        if is_torch:
            agent_flat = flat_state[..., : n_agents * AGENT_FEATURE_DIM]
            env_state = flat_state[..., n_agents * AGENT_FEATURE_DIM :]
            new_shape = list(flat_state.shape[:-1]) + [n_agents, AGENT_FEATURE_DIM]
            agent_states = agent_flat.reshape(*new_shape)
        else:
            agent_flat = flat_state[..., : n_agents * AGENT_FEATURE_DIM]
            env_state = flat_state[..., n_agents * AGENT_FEATURE_DIM :]
            new_shape = list(flat_state.shape[:-1]) + [n_agents, AGENT_FEATURE_DIM]
            agent_states = agent_flat.reshape(*new_shape)

        return agent_states, env_state


# ==============================================================================
# 2. Entity-Attention Permutation-Invariant Centralized Critic
# ==============================================================================

class EntityAttentionCritic(nn.Module):
    """
    Centralized Critic with Multi-Head Self-Attention and DeepSets Pooling.

    Key Features:
      - Permutation Invariance: The predicted collective value V(s) is strictly
        invariant to the ordering/permutation of drones in the input.
      - Fleet Size Agnostic: Evaluates arbitrary fleet counts (10, 25, 50, etc.)
        without resizing or retraining network parameters.
      - Individual Baselines: Evaluates both joint scalar baseline V(s)
        and per-agent baselines V_i(s) for counterfactual credit assignment.
    """

    def __init__(
        self,
        agent_feature_dim: int = AGENT_FEATURE_DIM,  # 12
        env_feature_dim: int = ENV_FEATURE_DIM,      # 11
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.agent_feature_dim = agent_feature_dim
        self.env_feature_dim = env_feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 1. Entity Feature Encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # 2. Inter-Agent Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(embed_dim)

        # 3. Environment Context Encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(env_feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Joint Representation: mean_pool (embed_dim) + max_pool (embed_dim) + env (64)
        joint_input_dim = 2 * embed_dim + 64

        # 4. Global Value Head V(s)
        self.joint_value_head = nn.Sequential(
            nn.Linear(joint_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 5. Agent-Specific Value Head V_i(s)
        self.individual_value_head = nn.Sequential(
            nn.Linear(joint_input_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self):
        """Orthogonal weight initialization with standard PPO gains."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Standard layers gain = sqrt(2)
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Value output layers initialized with gain = 1.0
        final_joint = self.joint_value_head[-1]
        nn.init.orthogonal_(final_joint.weight, gain=1.0)
        nn.init.constant_(final_joint.bias, 0.0)

        final_indiv = self.individual_value_head[-1]
        nn.init.orthogonal_(final_indiv.weight, gain=1.0)
        nn.init.constant_(final_indiv.bias, 0.0)

    def forward(
        self,
        agent_states: torch.Tensor,
        env_state: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward evaluation of the centralized critic.

        Can accept:
          - (agent_states, env_state)
          - OR flat_state passed as first argument (automatically unpacked)

        Args:
            agent_states: Tensor of shape (B, N, 12) or (N, 12) or flat (B, N*12 + 11)
            env_state: Tensor of shape (B, 11) or (11,) if not unpacked
            agent_mask: Optional boolean tensor of shape (B, N) or (N,)
            return_individual: If True, also returns (B, N) per-agent baselines

        Returns:
            v_joint: (B, 1) or scalar tensor representing V(s)
            v_agents (optional): (B, N) tensor representing V_i(s)
        """
        # Auto-unpack flat state if env_state is not provided
        if env_state is None and agent_states.dim() in (1, 2):
            last_dim = agent_states.shape[-1]
            if last_dim > AGENT_FEATURE_DIM:
                agent_states, env_state = GlobalStateBuilder.unpack_flat_state(agent_states)

        # Standardize batch dimensions
        unbatched = False
        if agent_states.dim() == 2:
            agent_states = agent_states.unsqueeze(0)  # (1, N, 12)
            if env_state is not None:
                env_state = env_state.unsqueeze(0)    # (1, 11)
            if agent_mask is not None:
                agent_mask = agent_mask.unsqueeze(0)  # (1, N)
            unbatched = True

        B, N, D_agent = agent_states.shape
        if env_state is None:
            env_state = torch.zeros((B, self.env_feature_dim), dtype=agent_states.dtype, device=agent_states.device)

        # 1. Encode Entity Features
        h = self.entity_encoder(agent_states)  # (B, N, embed_dim)

        # 2. Multi-Head Self-Attention
        # In PyTorch MHA, key_padding_mask is True for padded/ignored tokens
        key_padding_mask = ~agent_mask if agent_mask is not None else None
        attn_out, _ = self.self_attention(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )
        h = self.ln_attn(h + attn_out)  # Residual connection + norm: (B, N, embed_dim)

        # 3. Permutation-Invariant DeepSets Pooling
        if agent_mask is not None:
            mask_float = agent_mask.unsqueeze(-1).float()  # (B, N, 1)
            h_masked = h * mask_float
            counts = mask_float.sum(dim=1).clamp(min=1.0)
            mean_pool = h_masked.sum(dim=1) / counts       # (B, embed_dim)

            h_for_max = h.clone()
            h_for_max[~agent_mask] = -1e9
            max_pool = h_for_max.max(dim=1).values          # (B, embed_dim)
        else:
            mean_pool = h.mean(dim=1)                      # (B, embed_dim)
            max_pool = h.max(dim=1).values                 # (B, embed_dim)

        h_swarm = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2 * embed_dim)

        # 4. Encode Environment Context
        h_env = self.env_encoder(env_state)                # (B, 64)

        # 5. Joint Swarm Representation
        h_joint = torch.cat([h_swarm, h_env], dim=-1)      # (B, 2 * embed_dim + 64)

        # 6. Global Value Head V(s)
        v_joint = self.joint_value_head(h_joint)           # (B, 1)

        if not return_individual:
            if unbatched:
                return v_joint.squeeze(0)
            return v_joint

        # 7. Agent-Specific Value Head V_i(s)
        h_joint_expanded = h_joint.unsqueeze(1).expand(-1, N, -1)   # (B, N, joint_dim)
        agent_context = torch.cat([h_joint_expanded, h], dim=-1)     # (B, N, joint_dim + embed_dim)
        v_agents = self.individual_value_head(agent_context).squeeze(-1)  # (B, N)

        if unbatched:
            return v_joint.squeeze(0), v_agents.squeeze(0)
        return v_joint, v_agents


# ==============================================================================
# 3. MLP Centralized Critic Baseline
# ==============================================================================

class MLPCentralizedCritic(nn.Module):
    """
    Standard Feed-Forward Centralized Critic baseline for flattened global states.
    Used for benchmarks and comparisons against the entity-attention critic.
    """

    def __init__(
        self,
        input_dim: int = MAX_FLAT_STATE_DIM,
        hidden_dims: list[int] = [256, 256, 128],
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.value_head.bias is not None:
            nn.init.constant_(self.value_head.bias, 0.0)

    def forward(
        self,
        agent_states: torch.Tensor,
        env_state: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            agent_states: Tensor of shape (B, N, 12), (N, 12), or flat (B, input_dim) / (input_dim,)
            env_state: Optional Tensor of shape (B, 11) or (11,)

        Returns:
            Value scalar tensor of shape (B, 1) or (1,)
        """
        if env_state is not None:
            if agent_states.dim() == 2:
                # unbatched (N, 12) and (11,)
                ag_flat = agent_states.reshape(-1)
                curr_flat = torch.cat([ag_flat, env_state.reshape(-1)], dim=-1)
                if curr_flat.shape[0] < self.input_dim:
                    flat_state = torch.zeros(self.input_dim, dtype=curr_flat.dtype, device=curr_flat.device)
                    flat_state[:curr_flat.shape[0]] = curr_flat
                else:
                    flat_state = curr_flat[:self.input_dim]
            else:
                # batched (B, N, 12) and (B, 11)
                B = agent_states.shape[0]
                ag_flat = agent_states.reshape(B, -1)
                curr_flat = torch.cat([ag_flat, env_state], dim=-1)
                if curr_flat.shape[1] < self.input_dim:
                    flat_state = torch.zeros((B, self.input_dim), dtype=curr_flat.dtype, device=curr_flat.device)
                    flat_state[:, :curr_flat.shape[1]] = curr_flat
                else:
                    flat_state = curr_flat[:, :self.input_dim]
        else:
            flat_state = agent_states

        unbatched = False
        if flat_state.dim() == 1:
            flat_state = flat_state.unsqueeze(0)
            unbatched = True

        features = self.mlp(flat_state)
        v = self.value_head(features)

        if unbatched:
            return v.squeeze(0)
        return v


# ==============================================================================
# 4. Adaptive Value Normalization (PopArt)
# ==============================================================================

class ValueNorm(nn.Module):
    """
    Adaptive running mean and standard deviation normalization for value targets,
    with PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets) support.

    Under large multi-agent return scales (e.g., negative step penalties vs +100
    coverage bonuses), PopArt normalizes value targets for stable gradient descent
    while continuously rescaling the network's final output layer so that its
    unnormalized predictions remain mathematically invariant across statistic updates.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1,),
        beta: float = 0.995,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.register_buffer("running_mean", torch.zeros(input_shape, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(input_shape, dtype=torch.float32))
        self.register_buffer("running_std", torch.ones(input_shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def update(self, targets: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates running mean and variance using momentum-based running statistics.

        Args:
            targets: Batch of return targets

        Returns:
            Tuple of (new_running_mean, new_running_std)
        """
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets, dtype=torch.float32, device=self.running_mean.device)

        batch_mean = targets.mean()
        batch_var = targets.var(unbiased=False) if targets.numel() > 1 else torch.tensor(1.0, device=targets.device)

        if self.count == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp(min=self.epsilon))
        else:
            self.running_mean.copy_(self.beta * self.running_mean + (1.0 - self.beta) * batch_mean)
            self.running_var.copy_(self.beta * self.running_var + (1.0 - self.beta) * batch_var.clamp(min=self.epsilon))

        self.running_std.copy_(torch.sqrt(self.running_var + self.epsilon))
        self.count.add_(targets.numel())
        return self.running_mean.clone(), self.running_std.clone()

    def normalize(self, targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalizes targets: (y - mu) / sigma."""
        if isinstance(targets, np.ndarray):
            mu = float(self.running_mean.item())
            std = float(self.running_std.item())
            return (targets - mu) / max(std, self.epsilon)

        return (targets - self.running_mean) / self.running_std

    def denormalize(self, norm_values: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Denormalizes predictions: v * sigma + mu."""
        if isinstance(norm_values, np.ndarray):
            mu = float(self.running_mean.item())
            std = float(self.running_std.item())
            return norm_values * std + mu

        return norm_values * self.running_std + self.running_mean

    def update_popart(
        self,
        last_linear_layer: nn.Linear,
        targets: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs PopArt weight and bias adaptation.

        Updates running stats mu and sigma, then rescales:
          W_new = W_old * (sigma_old / sigma_new)
          b_new = (b_old * sigma_old + mu_old - mu_new) / sigma_new

        This guarantees that for any feature vector h:
          denormalize_new(W_new * h + b_new) == denormalize_old(W_old * h + b_old)
        """
        mu_old = self.running_mean.clone()
        std_old = self.running_std.clone()

        mu_new, std_new = self.update(targets)

        with torch.no_grad():
            weight = last_linear_layer.weight
            bias = last_linear_layer.bias

            std_ratio = std_old / std_new
            weight.mul_(std_ratio)

            if bias is not None:
                bias.copy_((bias * std_old + mu_old - mu_new) / std_new)

        return mu_new, std_new


# ==============================================================================
# 5. Multi-Agent Generalized Advantage Estimation (GAE)
# ==============================================================================

class MultiAgentGAE:
    """
    Computes Generalized Advantage Estimation (GAE) for multi-agent rollouts.

    Computes:
      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
      A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
      R_t     = A_t + V(s_t)
    """

    @staticmethod
    def compute(
        rewards: Union[np.ndarray, torch.Tensor],
        values: Union[np.ndarray, torch.Tensor],
        dones: Union[np.ndarray, torch.Tensor],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_adv: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Args:
            rewards: Shape (T, N) or (T, B, N)
            values:  Shape (T + 1, N) or (T + 1, B, N) containing bootstrap value at T
            dones:   Shape (T, N) or (T, B, N) indicating episode termination
            gamma:   Discount factor
            gae_lambda: GAE lambda parameter
            normalize_adv: If True, computes normalized advantages

        Returns:
            Dict containing:
              - 'advantages': Raw GAE advantages (T, ...)
              - 'returns': Target value returns R_t (T, ...)
              - 'normalized_advantages': Standardized advantages (T, ...)
        """
        is_torch = isinstance(rewards, torch.Tensor)
        T = rewards.shape[0]

        if is_torch:
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros_like(rewards[0])

            for t in reversed(range(T)):
                mask = 1.0 - dones[t].float()
                delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
                last_gae = delta + gamma * gae_lambda * mask * last_gae
                advantages[t] = last_gae

            returns = advantages + values[:T]

            if normalize_adv:
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False).clamp(min=1e-8)
                norm_adv = (advantages - adv_mean) / adv_std
            else:
                norm_adv = advantages.clone()

            return {
                "advantages": advantages,
                "returns": returns,
                "normalized_advantages": norm_adv,
            }

        # NumPy implementation
        rewards_np = np.asarray(rewards, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards_np)
        last_gae = np.zeros_like(rewards_np[0])

        for t in reversed(range(T)):
            mask = 1.0 - dones_np[t]
            delta = rewards_np[t] + gamma * values_np[t + 1] * mask - values_np[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np[:T]

        if normalize_adv:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            norm_adv = (advantages - adv_mean) / max(adv_std, 1e-8)
        else:
            norm_adv = advantages.copy()

        return {
            "advantages": advantages,
            "returns": returns,
            "normalized_advantages": norm_adv,
        }


# ==============================================================================
# 6. Centralized Critic Loss & Evaluation Utilities
# ==============================================================================

class CentralizedCriticLoss:
    """
    Computes value function loss with optional PPO-style value clipping:
      L_v = 0.5 * max((V - R)^2, (clip(V, V_old - eps, V_old + eps) - R)^2)
    """

    @staticmethod
    def compute_loss(
        v_pred: torch.Tensor,
        returns: torch.Tensor,
        v_old: Optional[torch.Tensor] = None,
        clip_param: Optional[float] = 0.2,
        use_huber: bool = False,
        huber_delta: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            v_pred: Current critic predictions (B, 1) or (B,)
            returns: Target returns (B, 1) or (B,)
            v_old: Previous policy rollout value baselines
            clip_param: PPO value clipping threshold epsilon
            use_huber: If True, uses Smooth L1 / Huber loss instead of MSE

        Returns:
            Scalar loss tensor
        """
        v_pred = v_pred.view_as(returns)

        if v_old is not None and clip_param is not None and clip_param > 0:
            v_old = v_old.view_as(returns)
            v_clipped = v_old + torch.clamp(v_pred - v_old, -clip_param, clip_param)
            loss_raw = (v_pred - returns) ** 2
            loss_clipped = (v_clipped - returns) ** 2
            loss = 0.5 * torch.max(loss_raw, loss_clipped).mean()
        elif use_huber:
            loss = F.smooth_l1_loss(v_pred, returns, beta=huber_delta)
        else:
            loss = 0.5 * F.mse_loss(v_pred, returns)

        return loss
