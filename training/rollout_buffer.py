"""
SwarmRL -- Day 17: Multi-Agent Rollout Buffer for MAPPO

Provides a trajectory replay buffer tailored for Centralized Training
with Decentralized Execution (CTDE):
  - Stores local agent observations (26 or 42 dims)
  - Stores centralized global environment states (agent entities + env context)
  - Stores actions, action log-probabilities, rewards, values, and dones
  - Computes Generalized Advantage Estimation (GAE) across multi-agent rollouts
  - Yields shuffled mini-batches for decentralized actor and centralized critic updates
"""

from __future__ import annotations

from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np
import torch

from training.centralized_critic import (
    MultiAgentGAE,
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
)


class MultiAgentRolloutBuffer:
    """
    Trajectory storage buffer for cooperative multi-agent PPO (MAPPO).

    Supports:
      - Variable fleet sizes N (e.g. 10, 25, 50 drones)
      - Rollout lengths T
      - GAE advantage estimation with optional advantage normalization
      - Shuffled mini-batch generators for actor and centralized critic training
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int = 3,
        agent_feature_dim: int = AGENT_FEATURE_DIM,  # 12
        env_feature_dim: int = ENV_FEATURE_DIM,      # 11
        device: str = "cpu",
    ):
        self.buffer_size = int(buffer_size)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.agent_feature_dim = int(agent_feature_dim)
        self.env_feature_dim = int(env_feature_dim)
        self.device = torch.device(device)

        # Storage buffers
        self.obs = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, act_dim), dtype=np.float32)
        self.action_log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=bool)

        # Centralized Critic Global States:
        # agent_states: (T, N, 12)
        # env_states:   (T, 11)
        self.agent_states = np.zeros(
            (buffer_size, num_agents, agent_feature_dim), dtype=np.float32
        )
        self.env_states = np.zeros((buffer_size, env_feature_dim), dtype=np.float32)

        # Values: (T + 1, 1) or (T + 1, N) for bootstrap
        self.values = np.zeros((buffer_size + 1, 1), dtype=np.float32)
        self.individual_values = np.zeros((buffer_size + 1, num_agents), dtype=np.float32)

        # Computed post-rollout
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.normalized_advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # Joint returns for centralized critic: (T, 1)
        self.joint_returns = np.zeros((buffer_size, 1), dtype=np.float32)
        self.joint_advantages = np.zeros((buffer_size, 1), dtype=np.float32)

        self.step = 0
        self.is_full = False

    def insert(
        self,
        obs: np.ndarray,
        agent_states: np.ndarray,
        env_state: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        rewards: np.ndarray,
        values: Union[float, np.ndarray],
        dones: np.ndarray,
        individual_values: Optional[np.ndarray] = None,
    ):
        """
        Inserts one simulation time-step transition for all agents into the buffer.

        Args:
            obs: (N, obs_dim) array of local agent observations
            agent_states: (N, 12) array of normalized agent entity states
            env_state: (11,) array of global environment context features
            actions: (N, act_dim) array of continuous actions
            action_log_probs: (N,) array of action log-probabilities
            rewards: (N,) array of agent rewards (or shared rewards broadcasted)
            values: scalar or (1,) centralized critic value baseline V(s_t)
            dones: (N,) array of termination/truncation flags
            individual_values: optional (N,) per-agent value baselines
        """
        if self.step >= self.buffer_size:
            raise IndexError(f"Buffer overflow: capacity is {self.buffer_size} steps.")

        t = self.step
        self.obs[t] = np.asarray(obs, dtype=np.float32)
        self.agent_states[t] = np.asarray(agent_states, dtype=np.float32)
        self.env_states[t] = np.asarray(env_state, dtype=np.float32)
        self.actions[t] = np.asarray(actions, dtype=np.float32)
        self.action_log_probs[t] = np.asarray(action_log_probs, dtype=np.float32)
        self.rewards[t] = np.asarray(rewards, dtype=np.float32)
        self.dones[t] = np.asarray(dones, dtype=bool)

        val_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        self.values[t] = val_arr[0] if len(val_arr) > 0 else 0.0

        if individual_values is not None:
            self.individual_values[t] = np.asarray(individual_values, dtype=np.float32)
        else:
            self.individual_values[t] = float(self.values[t, 0])

        self.step += 1
        if self.step == self.buffer_size:
            self.is_full = True

    def compute_returns_and_advantages(
        self,
        last_value: Union[float, np.ndarray],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_adv: bool = True,
        last_individual_values: Optional[np.ndarray] = None,
    ):
        """
        Computes multi-agent GAE advantages and target returns using MultiAgentGAE.

        Args:
            last_value: Bootstrap value baseline V(s_T) at terminal step
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_adv: If True, computes standardized advantages
            last_individual_values: Bootstrap individual baselines (N,)
        """
        T = self.step
        if T == 0:
            return

        # Store bootstrap values at index T
        val_arr = np.asarray(last_value, dtype=np.float32).reshape(-1)
        self.values[T] = val_arr[0] if len(val_arr) > 0 else 0.0

        if last_individual_values is not None:
            self.individual_values[T] = np.asarray(last_individual_values, dtype=np.float32)
        else:
            self.individual_values[T] = float(self.values[T, 0])

        # 1. Agent-level GAE calculation
        rewards_slice = self.rewards[:T]  # (T, N)
        values_slice = self.individual_values[: T + 1]  # (T + 1, N)
        dones_slice = self.dones[:T]  # (T, N)

        agent_gae = MultiAgentGAE.compute(
            rewards=rewards_slice,
            values=values_slice,
            dones=dones_slice,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_adv=normalize_adv,
        )

        self.advantages[:T] = agent_gae["advantages"]
        self.returns[:T] = agent_gae["returns"]
        self.normalized_advantages[:T] = agent_gae["normalized_advantages"]

        # 2. Joint Centralized Critic GAE calculation (using mean collective reward)
        joint_rewards = np.mean(rewards_slice, axis=-1, keepdims=True)  # (T, 1)
        joint_values = self.values[: T + 1]  # (T + 1, 1)
        joint_dones = np.any(dones_slice, axis=-1, keepdims=True)  # (T, 1)

        joint_gae = MultiAgentGAE.compute(
            rewards=joint_rewards,
            values=joint_values,
            dones=joint_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_adv=False,
        )

        self.joint_advantages[:T] = joint_gae["advantages"]
        self.joint_returns[:T] = joint_gae["returns"]

    def actor_mini_batch_generator(
        self,
        mini_batch_size: int,
        num_epochs: int = 5,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields shuffled mini-batches for decentralized actor policy optimization.
        Flattens (T, N) trajectory samples into total_samples = T * N agent transitions.
        """
        T = self.step
        N = self.num_agents
        total_samples = T * N

        # Flatten along time and agent dimensions
        flat_obs = self.obs[:T].reshape(total_samples, self.obs_dim)
        flat_actions = self.actions[:T].reshape(total_samples, self.act_dim)
        flat_log_probs = self.action_log_probs[:T].reshape(total_samples)
        flat_advantages = self.normalized_advantages[:T].reshape(total_samples)
        flat_returns = self.returns[:T].reshape(total_samples)

        for _ in range(num_epochs):
            indices = np.random.permutation(total_samples)
            for start_idx in range(0, total_samples, mini_batch_size):
                batch_indices = indices[start_idx : start_idx + mini_batch_size]

                yield {
                    "obs": torch.as_tensor(
                        flat_obs[batch_indices], dtype=torch.float32, device=self.device
                    ),
                    "actions": torch.as_tensor(
                        flat_actions[batch_indices], dtype=torch.float32, device=self.device
                    ),
                    "old_log_probs": torch.as_tensor(
                        flat_log_probs[batch_indices], dtype=torch.float32, device=self.device
                    ),
                    "advantages": torch.as_tensor(
                        flat_advantages[batch_indices], dtype=torch.float32, device=self.device
                    ),
                    "returns": torch.as_tensor(
                        flat_returns[batch_indices], dtype=torch.float32, device=self.device
                    ),
                }

    def critic_mini_batch_generator(
        self,
        mini_batch_size: int,
        num_epochs: int = 5,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields shuffled mini-batches of global state snapshots (T samples)
        for centralized critic value function optimization.
        """
        T = self.step
        for _ in range(num_epochs):
            indices = np.random.permutation(T)
            for start_idx in range(0, T, mini_batch_size):
                batch_indices = indices[start_idx : start_idx + mini_batch_size]

                yield {
                    "agent_states": torch.as_tensor(
                        self.agent_states[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "env_state": torch.as_tensor(
                        self.env_states[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "target_returns": torch.as_tensor(
                        self.joint_returns[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "old_values": torch.as_tensor(
                        self.values[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                }

    def clear(self):
        """Clears the buffer pointers for the next training iteration."""
        self.step = 0
        self.is_full = False
