"""
SwarmRL -- Day 17: MAPPO Actor & Policy Architecture

Implements decentralized actor networks and combined actor-critic policies:
  - MAPPOActor: Multi-layer continuous Gaussian policy with LayerNorm,
    tanh-bounded action means, learned log-stds, and Gaussian entropy.
  - MAPPOPolicy: Cohesive wrapper binding decentralized actor and centralized critic
    for seamless rollout stepping, evaluation, and inference.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from training.centralized_critic import (
    EntityAttentionCritic,
    MLPCentralizedCritic,
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
)


class MAPPOActor(nn.Module):
    """
    Decentralized actor policy network.

    Maps a single drone's local observation (26-dim base or 42-dim comm-augmented)
    to continuous 3D acceleration actions [ax, ay, az] in [-1, 1].
    """

    def __init__(
        self,
        obs_dim: int = 42,
        act_dim: int = 3,
        hidden_dims: list[int] = [256, 256, 128],
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Feature Trunk
        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.trunk = nn.Sequential(*layers)

        # Mean Head: tanh-scaled to [-1, 1]
        self.action_mean = nn.Sequential(
            nn.Linear(prev_dim, act_dim),
            nn.Tanh(),
        )

        # Log standard deviation parameter
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std, dtype=torch.float32))

        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self):
        """Orthogonal weight initialization adhering to standard PPO best practices."""
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Output linear layer initialized with small gain (0.01) for smooth initial actions
        mean_linear = self.action_mean[0]
        if isinstance(mean_linear, nn.Linear):
            nn.init.orthogonal_(mean_linear.weight, gain=0.01)
            if mean_linear.bias is not None:
                nn.init.constant_(mean_linear.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing mean and standard deviation.

        Args:
            obs: (..., obs_dim) observation tensor

        Returns:
            Tuple of (mean, std)
        """
        features = self.trunk(obs)
        mean = self.action_mean(features)
        clamped_log_std = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = torch.exp(clamped_log_std).expand_as(mean)
        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action given an observation and computes its log-probability.

        Args:
            obs: Tensor of shape (..., obs_dim)
            deterministic: If True, uses the mean directly without sampling

        Returns:
            Tuple of (action, log_prob)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log-probabilities and distribution entropy for given state-action pairs.

        Args:
            obs: Tensor of shape (B, obs_dim)
            actions: Tensor of shape (B, act_dim)

        Returns:
            Tuple of (log_probs, entropy)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        # Sum log probs across action dimensions
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class MAPPOPolicy(nn.Module):
    """
    Cohesive CTDE Multi-Agent PPO Policy.
    Combines a decentralized actor with a centralized critic.
    """

    def __init__(
        self,
        obs_dim: int = 42,
        act_dim: int = 3,
        agent_feature_dim: int = AGENT_FEATURE_DIM,  # 12
        env_feature_dim: int = ENV_FEATURE_DIM,      # 11
        critic_type: str = "attention",              # "attention" or "mlp"
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dims: list[int] = [256, 256, 128],
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.critic_type = critic_type

        # 1. Decentralized Actor
        self.actor = MAPPOActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
        )

        # 2. Centralized Critic
        if critic_type == "attention":
            self.critic = EntityAttentionCritic(
                agent_feature_dim=agent_feature_dim,
                env_feature_dim=env_feature_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        else:
            self.critic = MLPCentralizedCritic()

        self.to(self.device)

    def act(
        self,
        obs: np.ndarray,
        agent_states: np.ndarray,
        env_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Step-time policy evaluation during rollouts.

        Args:
            obs: (N, obs_dim) array of local agent observations
            agent_states: (N, 12) array of normalized agent features
            env_state: (11,) array of global environment context
            deterministic: If True, uses mean action without noise

        Returns:
            Tuple of:
              - actions: (N, 3) numpy array
              - action_log_probs: (N,) numpy array
              - value: scalar float centralized value baseline V(s)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions_t, log_probs_t = self.actor.get_action(obs_t, deterministic=deterministic)

            ag_t = torch.as_tensor(agent_states, dtype=torch.float32, device=self.device)
            env_t = torch.as_tensor(env_state, dtype=torch.float32, device=self.device)
            val_t = self.critic(ag_t, env_t)

        return (
            actions_t.cpu().numpy(),
            log_probs_t.cpu().numpy(),
            float(val_t.item()),
        )

    def evaluate_critic(
        self,
        agent_states: Union[np.ndarray, torch.Tensor],
        env_state: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluates centralized critic V(s)."""
        if not isinstance(agent_states, torch.Tensor):
            agent_states = torch.as_tensor(agent_states, dtype=torch.float32, device=self.device)
        if not isinstance(env_state, torch.Tensor):
            env_state = torch.as_tensor(env_state, dtype=torch.float32, device=self.device)
        return self.critic(agent_states, env_state)
