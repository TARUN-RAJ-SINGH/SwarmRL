"""
SwarmRL -- Day 17: Full MAPPO CTDE Training Engine

Implements the multi-agent proximal policy optimization training loop
with Centralized Training and Decentralized Execution (CTDE):
  - MAPPOConfig: Structured training hyperparameter container
  - MAPPOTrainer: Complete multi-agent trainer handling rollouts,
    PPO actor surrogate optimization, centralized critic updates,
    PopArt target rescaling, gradient clipping, diagnostics, and checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from training.rollout_buffer import MultiAgentRolloutBuffer
from training.mappo_policy import MAPPOPolicy
from training.centralized_critic import (
    GlobalStateBuilder,
    ValueNorm,
    CentralizedCriticLoss,
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
)


@dataclass
class MAPPOConfig:
    """Hyperparameter specification for MAPPO training."""
    # Optimization
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    eps_adam: float = 1e-5
    max_grad_norm: float = 0.5

    # PPO Parameters
    clip_param: float = 0.2
    ppo_epochs: int = 5
    actor_mini_batch_size: int = 64
    critic_mini_batch_size: int = 32
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5

    # GAE Parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_adv: bool = True

    # Critic Options
    critic_type: str = "attention"  # "attention" or "mlp"
    use_popart: bool = True
    use_clipped_value_loss: bool = True
    clip_value_loss_param: float = 0.2
    use_huber_loss: bool = True
    huber_delta: float = 1.0

    # Architecture
    embed_dim: int = 128
    num_heads: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])


class MAPPOTrainer:
    """
    Cooperative Multi-Agent PPO Trainer with Centralized Critic (CTDE).
    """

    def __init__(
        self,
        config: MAPPOConfig,
        obs_dim: int = 42,
        num_agents: int = 10,
        rollout_steps: int = 200,
        device: str = "cpu",
    ):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.num_agents = int(num_agents)
        self.rollout_steps = int(rollout_steps)
        self.device = torch.device(device)

        # 1. Policy (Decentralized Actor + Centralized Critic)
        self.policy = MAPPOPolicy(
            obs_dim=obs_dim,
            act_dim=3,
            critic_type=config.critic_type,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            hidden_dims=config.hidden_dims,
            device=device,
        )

        # 2. Value Normalization (PopArt)
        self.value_norm = ValueNorm() if config.use_popart else None

        # 3. Rollout Buffer
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=rollout_steps,
            num_agents=num_agents,
            obs_dim=obs_dim,
            act_dim=3,
            device=device,
        )

        # 4. Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=config.lr_actor,
            eps=config.eps_adam,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=config.lr_critic,
            eps=config.eps_adam,
        )

        self.iteration = 0
        self.state_builder = GlobalStateBuilder()

    def collect_rollout(
        self,
        env: Any,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the multi-drone simulation for num_steps, collecting transitions
        and inserting them into the rollout buffer.

        Args:
            env: SwarmEnv or CommSwarmEnv instance
            num_steps: Rollout length (defaults to self.rollout_steps)

        Returns:
            Dictionary of rollout performance metrics (rewards, coverage, collisions)
        """
        steps_to_run = num_steps if num_steps is not None else self.rollout_steps
        self.buffer.clear()

        # Check if environment is already active or needs reset
        inner = getattr(env, "_inner", env)
        if inner.current_step == 0 or len(env.agents) == 0:
            obs_dict, _ = env.reset()
        else:
            # Query current observations from inner step result
            res = inner.last_step_result
            obs_dict = {f"drone_{did}": o for did, o in res["observations"].items()}
            # If wrapped by CommSwarmEnv, enrich
            if hasattr(env, "_build_comm_obs"):
                obs_dict = env._build_comm_obs(obs_dict)

        total_reward = 0.0
        step_rewards = []
        episode_collisions = 0

        for t in range(steps_to_run):
            # 1. Format local observations array: (N, obs_dim)
            obs_array = np.array(
                [obs_dict.get(f"drone_{i}", np.zeros(self.obs_dim)) for i in range(self.num_agents)],
                dtype=np.float32,
            )

            # 2. Extract global state
            global_state = env.get_global_state()
            ag_states = global_state["agent_states"]
            env_state = global_state["env_state"]

            # 3. Policy Action Inference & Centralized Value Prediction
            actions, log_probs, value = self.policy.act(
                obs=obs_array,
                agent_states=ag_states,
                env_state=env_state,
                deterministic=False,
            )

            # 4. Step Environment
            action_dict = {f"drone_{i}": actions[i] for i in range(self.num_agents)}
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            rewards_array = np.array(
                [rewards.get(f"drone_{i}", 0.0) for i in range(self.num_agents)],
                dtype=np.float32,
            )
            dones_array = np.array(
                [terminations.get(f"drone_{i}", False) or truncations.get(f"drone_{i}", False)
                 for i in range(self.num_agents)],
                dtype=bool,
            )

            # 5. Insert transition into buffer
            self.buffer.insert(
                obs=obs_array,
                agent_states=ag_states,
                env_state=env_state,
                actions=actions,
                action_log_probs=log_probs,
                rewards=rewards_array,
                values=value,
                dones=dones_array,
            )

            # Track metrics
            mean_step_rew = float(np.mean(rewards_array))
            total_reward += mean_step_rew
            step_rewards.append(mean_step_rew)

            for info in infos.values():
                episode_collisions += info.get("collisions", 0)

            obs_dict = next_obs

            # If all agents terminated or truncated, reset
            if len(env.agents) == 0:
                obs_dict, _ = env.reset()

        # 6. Bootstrap value at step T for GAE
        final_global = env.get_global_state()
        final_ag = final_global["agent_states"]
        final_env = final_global["env_state"]
        with torch.no_grad():
            bootstrap_val = self.policy.evaluate_critic(final_ag, final_env).item()

        # 7. Compute GAE Returns and Advantages
        self.buffer.compute_returns_and_advantages(
            last_value=bootstrap_val,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_adv=self.config.normalize_adv,
        )

        return {
            "mean_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "total_reward": float(total_reward),
            "final_coverage": float(final_global.get("coverage", 0.0)),
            "zone_a_coverage": float(final_global.get("zone_a_coverage", 0.0)),
            "zone_b_coverage": float(final_global.get("zone_b_coverage", 0.0)),
            "collisions": int(episode_collisions),
        }

    def train_step(self) -> Dict[str, float]:
        """
        Executes PPO update epochs over the collected rollout buffer:
          - Optimizes decentralized actor policy with clipped surrogate loss & entropy
          - Optimizes centralized critic with (clipped) value loss & PopArt targets
        """
        cfg = self.config
        T = self.buffer.step
        if T == 0:
            return {}

        # 1. Update ValueNorm statistics with joint returns if PopArt enabled
        if self.value_norm is not None:
            joint_returns = self.buffer.joint_returns[:T]
            self.value_norm.update(joint_returns)

        # Metrics trackers
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []

        # 2. PPO Actor Optimization Epochs
        actor_gen = self.buffer.actor_mini_batch_generator(
            mini_batch_size=cfg.actor_mini_batch_size,
            num_epochs=cfg.ppo_epochs,
        )

        for batch in actor_gen:
            obs_b = batch["obs"]
            actions_b = batch["actions"]
            old_log_probs_b = batch["old_log_probs"]
            advantages_b = batch["advantages"]

            # Evaluate actions under current policy
            log_probs, entropy = self.policy.actor.evaluate_actions(obs_b, actions_b)

            # Ratio r(theta) = exp(log_prob - old_log_prob)
            ratio = torch.exp(log_probs - old_log_probs_b)

            # Clipped surrogate loss
            surr1 = ratio * advantages_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages_b
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy_loss = -cfg.entropy_coef * entropy.mean()

            total_actor_loss = actor_loss + entropy_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), cfg.max_grad_norm)
            self.actor_optimizer.step()

            # Metrics
            policy_losses.append(actor_loss.item())
            entropies.append(entropy.mean().item())

            with torch.no_grad():
                approx_kl = 0.5 * torch.mean((old_log_probs_b - log_probs) ** 2).item()
                approx_kls.append(approx_kl)
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_param).float()).item()
                clip_fractions.append(clip_frac)

        # 3. Centralized Critic Optimization Epochs
        critic_gen = self.buffer.critic_mini_batch_generator(
            mini_batch_size=cfg.critic_mini_batch_size,
            num_epochs=cfg.ppo_epochs,
        )

        for batch in critic_gen:
            ag_b = batch["agent_states"]
            env_b = batch["env_state"]
            targets_b = batch["target_returns"]
            old_val_b = batch["old_values"]

            # Normalize target returns with PopArt if enabled
            if self.value_norm is not None:
                norm_targets = self.value_norm.normalize(targets_b)
                norm_old_val = self.value_norm.normalize(old_val_b)
            else:
                norm_targets = targets_b
                norm_old_val = old_val_b

            # Critic prediction
            v_pred = self.policy.evaluate_critic(ag_b, env_b)

            # Critic loss computation
            critic_loss = CentralizedCriticLoss.compute_loss(
                v_pred=v_pred,
                returns=norm_targets,
                v_old=norm_old_val if cfg.use_clipped_value_loss else None,
                clip_param=cfg.clip_value_loss_param if cfg.use_clipped_value_loss else None,
                use_huber=cfg.use_huber_loss,
                huber_delta=cfg.huber_delta,
            )

            total_critic_loss = cfg.value_loss_coef * critic_loss

            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()

            value_losses.append(critic_loss.item())

        # 4. Compute Explained Variance
        all_targets = self.buffer.joint_returns[:T].flatten()
        all_old_vals = self.buffer.values[:T].flatten()
        var_targets = np.var(all_targets)
        if var_targets > 1e-8:
            explained_var = float(1.0 - np.var(all_targets - all_old_vals) / var_targets)
        else:
            explained_var = 0.0

        self.iteration += 1

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "explained_variance": explained_var,
        }

    def train_iteration(self, env: Any, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes a complete training cycle:
          1. Collects multi-agent rollout from simulation
          2. Runs PPO optimization over actor and critic
          3. Returns merged telemetry and training diagnostics
        """
        rollout_stats = self.collect_rollout(env, num_steps=num_steps)
        train_stats = self.train_step()

        merged = {
            "iteration": self.iteration,
            **rollout_stats,
            **train_stats,
        }
        return merged

    def save_checkpoint(self, filepath: str):
        """Saves policy weights, optimizer states, and training metadata."""
        import dataclasses
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "iteration": self.iteration,
            "actor_state_dict": self.policy.actor.state_dict(),
            "critic_state_dict": self.policy.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "value_norm_state_dict": self.value_norm.state_dict() if self.value_norm else None,
            "config": dataclasses.asdict(self.config),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Restores policy weights and training states from a checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        self.iteration = checkpoint.get("iteration", 0)
        self.policy.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.policy.critic.load_state_dict(checkpoint["critic_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if self.value_norm and checkpoint.get("value_norm_state_dict"):
            self.value_norm.load_state_dict(checkpoint["value_norm_state_dict"])
