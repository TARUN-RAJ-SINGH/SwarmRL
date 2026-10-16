"""
SwarmRL -- Day 15: GNN Communication Architecture

Implements a Graph Neural Network message-passing layer for inter-agent
communication. Each drone broadcasts a compact message to its neighbours
and aggregates incoming messages via spatial attention to form a richer
cooperative observation.

Architecture
------------
  Message vector  : 16-dim per neighbour
    [0-2]  relative position (dx, dy, dz) normalised
    [3-5]  neighbour velocity (vx, vy, vz) normalised
    [6]    neighbour speed fraction
    [7-11] 5 representative LiDAR values (sampled from 10 rays)
    [12]   neighbour local coverage fraction
    [13]   neighbour global coverage fraction
    [14]   neighbour heading normalised
    [15]   normalised inter-drone distance

  Attention weight: softmax over inverse distances within comm range

  Aggregated context: weighted sum of K neighbour messages → 16-dim vector

  Enhanced observation: cat(base_obs[26], context[16]) → 42-dim

  GNN policy MLP: 42 → [256, 256, 128] → action 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MSG_DIM = 16          # message vector dimension per neighbour
BASE_OBS_DIM = 26     # from Day 3 spaces.py
ENHANCED_OBS_DIM = BASE_OBS_DIM + MSG_DIM   # 42

# Max neighbours used in attention aggregation
MAX_NEIGHBOURS = 9    # up to N-1 drones; we pad to this

COMM_RANGE_NORM = 1.0  # normalised: distances already in [0,1] by sensor_range


# ─────────────────────────────────────────────────────────────
# Message Encoder  (pure NumPy, runs inside env step)
# ─────────────────────────────────────────────────────────────

class MessageEncoder:
    """
    Extracts a 16-dim message from a drone's base observation and the
    relative position to the receiver drone.

    Operates entirely in NumPy so it can run inside the PettingZoo env
    step() without GPU overhead.
    """

    # Indices into the 26-dim base observation
    # [0-2] pos_norm, [3-5] vel_norm, [6] speed_frac, [11-20] LiDAR (10 rays)
    # [21] local_cov, [22] global_cov, [25] heading_norm

    LIDAR_INDICES = list(range(11, 21))     # 10 LiDAR values
    LIDAR_SAMPLE  = [11, 13, 15, 17, 19]   # 5 representative rays

    def encode(
        self,
        sender_obs: np.ndarray,
        receiver_pos_norm: np.ndarray,
        dist_norm: float,
    ) -> np.ndarray:
        """
        Build a 16-dim message from sender's observation relative to receiver.

        Args:
            sender_obs:       26-dim observation of the sending drone
            receiver_pos_norm: 3-dim normalised position of the receiver
            dist_norm:        normalised distance sender→receiver in [0,1]

        Returns:
            np.ndarray of shape (MSG_DIM,) dtype float32
        """
        # Relative position (sender_pos - receiver_pos), already normalised
        sender_pos_norm = sender_obs[0:3]
        rel_pos = np.clip(sender_pos_norm - receiver_pos_norm, -1.0, 1.0)

        # Sender velocity normalised (already in [-1,1] per obs definition)
        vel_norm = sender_obs[3:6]

        # Sender speed fraction
        speed_frac = np.array([sender_obs[6]], dtype=np.float32)

        # 5 sampled LiDAR rays
        lidar_sample = sender_obs[self.LIDAR_SAMPLE].astype(np.float32)

        # Coverage fracs
        local_cov  = np.array([sender_obs[21]], dtype=np.float32)
        global_cov = np.array([sender_obs[22]], dtype=np.float32)

        # Heading normalised
        heading = np.array([sender_obs[25]], dtype=np.float32)

        # Distance (already normalised by caller)
        dist = np.array([float(dist_norm)], dtype=np.float32)

        msg = np.concatenate([
            rel_pos,        # 3
            vel_norm,       # 3
            speed_frac,     # 1
            lidar_sample,   # 5
            local_cov,      # 1
            global_cov,     # 1
            heading,        # 1
            dist,           # 1
        ]).astype(np.float32)                 # total = 16

        assert msg.shape == (MSG_DIM,), f"Msg shape {msg.shape} != ({MSG_DIM},)"
        return msg


# ─────────────────────────────────────────────────────────────
# Spatial Attention Aggregator  (NumPy)
# ─────────────────────────────────────────────────────────────

class SpatialAttentionAggregator:
    """
    Aggregates a variable-length list of neighbour messages into a fixed
    MSG_DIM context vector using distance-based softmax attention.

    Attention weight for neighbour i:
        a_i = softmax( temperature / dist_i )

    Context = sum_i( a_i * msg_i )
    """

    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: Scales the inverse-distance logits.
                         Higher = sharper focus on nearest neighbour.
        """
        self.temperature = temperature

    def aggregate(
        self,
        messages: List[np.ndarray],
        distances: List[float],
    ) -> np.ndarray:
        """
        Aggregate neighbour messages into a single context vector.

        Args:
            messages:  List of (MSG_DIM,) arrays from neighbours
            distances: List of normalised distances (same order as messages)

        Returns:
            np.ndarray of shape (MSG_DIM,) — aggregated context
        """
        if len(messages) == 0:
            return np.zeros(MSG_DIM, dtype=np.float32)

        msgs  = np.stack(messages, axis=0)           # (K, 16)
        dists = np.array(distances, dtype=np.float32) # (K,)

        # Compute attention logits: higher weight for closer drones
        eps = 1e-6
        logits = self.temperature / (dists + eps)     # (K,)

        # Softmax
        logits -= logits.max()                         # numerical stability
        weights = np.exp(logits)
        weights /= (weights.sum() + eps)               # (K,)

        # Weighted sum
        context = (weights[:, None] * msgs).sum(axis=0)  # (MSG_DIM,)
        return context.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# CommNet / GNN Layer  (PyTorch Module — used inside policy)
# ─────────────────────────────────────────────────────────────

class GNNCommLayer(nn.Module):
    """
    A single graph message-passing layer for multi-agent communication.

    Operates on a batch of per-agent enhanced observations:
        input: (batch, num_agents, ENHANCED_OBS_DIM)

    Internally:
      1. Projects each agent's base obs to a hidden key/value.
      2. Computes pairwise attention using the distance encoded in obs[15].
      3. Aggregates neighbour values.
      4. Returns a refined per-agent embedding of dimension hidden_dim.

    This layer is used as the first layer of the GNN policy network to
    re-learn communication-aware representations end-to-end during training.
    """

    def __init__(
        self,
        input_dim:  int = ENHANCED_OBS_DIM,
        hidden_dim: int = 128,
        msg_dim:    int = MSG_DIM,
        num_heads:  int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        # Encoder: maps full observation → query, key, value
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)

        # Output projection after attention
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalisation for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        comm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GNN communication layer.

        Args:
            x:         (batch, num_agents, input_dim) — enhanced observations
            comm_mask: (batch, num_agents, num_agents) boolean — True where
                       communication is allowed. If None, all-to-all.

        Returns:
            (batch, num_agents, hidden_dim) — attended agent embeddings
        """
        B, N, _ = x.shape

        # Project to Q, K, V  — (B, N, hidden_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: (B, N, num_heads, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        # Transpose: (B, num_heads, N, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention: (B, num_heads, N, N)
        scale = float(self.head_dim) ** -0.5
        attn  = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply communication mask (mask out non-neighbours)
        if comm_mask is not None:
            # comm_mask: (B, N, N) → expand to (B, num_heads, N, N)
            mask = comm_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Weighted sum: (B, num_heads, N, head_dim)
        out = torch.matmul(attn, V)

        # Merge heads: (B, N, hidden_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        # Output projection + residual norm
        out = self.out_proj(out)
        # Simple residual: project x to hidden_dim first if needed
        out = self.layer_norm(out)

        return out


# ─────────────────────────────────────────────────────────────
# GNN Policy Network  (standalone PyTorch Module)
# ─────────────────────────────────────────────────────────────

class GNNPolicyNetwork(nn.Module):
    """
    Full actor-critic network with GNN communication pre-processing.

    Architecture:
        GNNCommLayer(42 → 128) → MLP [256, 256, 128] → [actor head | critic head]

    Designed to be wrapped inside a custom RLlib RLModule.
    Can also be used standalone for evaluation.
    """

    def __init__(
        self,
        obs_dim:    int = ENHANCED_OBS_DIM,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # GNN communication layer
        self.gnn_layer = GNNCommLayer(
            input_dim  = obs_dim,
            hidden_dim = hidden_dim,
            num_heads  = 4,
        )

        # Shared MLP trunk (processes GNN output per agent)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head: outputs action mean
        self.actor_mean = nn.Linear(128, action_dim)
        # Learned log-std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs state value
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self,
        obs: torch.Tensor,
        comm_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs:       (batch, num_agents, obs_dim) OR (batch, obs_dim) for
                       single-agent RLlib calls. Shape is auto-detected.
            comm_mask: (batch, num_agents, num_agents) boolean comm mask.

        Returns:
            action_mean: (batch, [num_agents,] action_dim)
            log_std:     (action_dim,) broadcast compatible
            value:       (batch, [num_agents,] 1)
        """
        single = (obs.dim() == 2)
        if single:
            # RLlib calls with (B, obs_dim) — treat as single agent
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)

        # GNN pass: (B, N, hidden_dim)
        h = self.gnn_layer(obs, comm_mask)

        # MLP trunk per agent: (B, N, 128)
        h = self.trunk(h)

        action_mean = torch.tanh(self.actor_mean(h))  # (B, N, 3)  in (-1,1)
        value       = self.critic(h)                   # (B, N, 1)

        if single:
            action_mean = action_mean.squeeze(1)  # (B, 3)
            value       = value.squeeze(1)        # (B, 1)

        return action_mean, self.log_std, value


# ─────────────────────────────────────────────────────────────
# Bandwidth / Packet-Loss Resilience Utilities
# ─────────────────────────────────────────────────────────────

def apply_packet_loss(
    messages: List[np.ndarray],
    distances: List[float],
    loss_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Simulate communication packet loss by randomly dropping messages.

    Args:
        messages:   List of neighbour messages
        distances:  Corresponding distances
        loss_prob:  Probability of dropping any single message [0, 1]
        rng:        Optional random generator for reproducibility

    Returns:
        Filtered (messages, distances) with some packets dropped
    """
    if loss_prob <= 0.0 or len(messages) == 0:
        return messages, distances

    rng = rng or np.random.default_rng()
    kept_msgs  = []
    kept_dists = []
    for msg, dist in zip(messages, distances):
        if rng.random() > loss_prob:
            kept_msgs.append(msg)
            kept_dists.append(dist)
    return kept_msgs, kept_dists


def apply_bandwidth_limit(
    messages:  List[np.ndarray],
    distances: List[float],
    max_msgs:  int = 4,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Enforce a bandwidth cap by keeping only the max_msgs nearest neighbours.

    Args:
        messages:  Full list of neighbour messages
        distances: Corresponding distances
        max_msgs:  Maximum messages to retain

    Returns:
        Filtered (messages, distances) capped to max_msgs nearest
    """
    if len(messages) <= max_msgs:
        return messages, distances

    paired = sorted(zip(distances, messages), key=lambda x: x[0])
    paired = paired[:max_msgs]
    dists, msgs = zip(*paired)
    return list(msgs), list(dists)


# ─────────────────────────────────────────────────────────────
# Build Comm Observation  (called in env step)
# ─────────────────────────────────────────────────────────────

_encoder     = MessageEncoder()
_aggregator  = SpatialAttentionAggregator(temperature=2.0)

def build_comm_observation(
    drone_id:        int,
    base_obs:        np.ndarray,
    all_obs:         Dict[int, np.ndarray],
    drone_positions: Dict[int, np.ndarray],
    comm_range:      float = 20.0,
    sensor_range:    float = 10.0,
    loss_prob:       float = 0.0,
    max_msgs:        int   = 9,
) -> np.ndarray:
    """
    Build an enhanced (42-dim) observation for one drone by augmenting
    its 26-dim base observation with a 16-dim aggregated comm context.

    Args:
        drone_id:        ID of the receiving drone
        base_obs:        26-dim observation of the receiver
        all_obs:         Dict[drone_id → 26-dim obs] for ALL drones
        drone_positions: Dict[drone_id → 3-dim world position]
        comm_range:      Communication range in metres
        sensor_range:    Sensor range for normalisation
        loss_prob:       Packet-loss probability [0, 1]
        max_msgs:        Max neighbours to aggregate (bandwidth cap)

    Returns:
        np.ndarray of shape (ENHANCED_OBS_DIM,) = (42,)
    """
    receiver_pos      = drone_positions[drone_id]
    receiver_pos_norm = base_obs[0:3]   # already normalised

    messages  = []
    distances = []

    for sender_id, sender_obs in all_obs.items():
        if sender_id == drone_id:
            continue

        sender_pos = drone_positions[sender_id]
        dist_m     = float(np.linalg.norm(receiver_pos - sender_pos))

        if dist_m > comm_range:
            continue  # out of range

        dist_norm = float(np.clip(dist_m / sensor_range, 0.0, 1.0))

        msg = _encoder.encode(sender_obs, receiver_pos_norm, dist_norm)
        messages.append(msg)
        distances.append(dist_norm)

    # Bandwidth cap
    messages, distances = apply_bandwidth_limit(messages, distances, max_msgs)

    # Packet loss simulation
    if loss_prob > 0.0:
        messages, distances = apply_packet_loss(messages, distances, loss_prob)

    # Aggregate
    context = _aggregator.aggregate(messages, distances)

    # Concatenate base obs + context
    enhanced = np.concatenate([base_obs, context]).astype(np.float32)
    assert enhanced.shape == (ENHANCED_OBS_DIM,), (
        f"Enhanced obs shape {enhanced.shape} != ({ENHANCED_OBS_DIM},)"
    )
    return enhanced
