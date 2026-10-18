"""
SwarmRL -- Day 17: MAPPO CTDE Tests

Comprehensive test suite covering:
  Suite 1: Multi-Agent Rollout Buffer (Storage, GAE computation, mini-batch generators)
  Suite 2: Decentralized Actor & MAPPO Policy (Gaussian distribution, actions, log_probs, entropy)
  Suite 3: MAPPO Loss Formulations (PPO clipped surrogate, KL divergence, entropy bonus)
  Suite 4: MAPPOTrainer Training Loop & Checkpointing (Rollouts, optimization, save/load)
  Suite 5: Fleet Scalability & Multi-Agent Integration (10 & 25 drones with CommSwarmEnv and SwarmEnv)
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

# ── path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.world_config import WORLD_CONFIG
from env.spaces import OBS_DIM
from env.gnn_comm import ENHANCED_OBS_DIM
from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from training.centralized_critic import (
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
    GlobalStateBuilder,
    ValueNorm,
)
from training.rollout_buffer import MultiAgentRolloutBuffer
from training.mappo_policy import MAPPOActor, MAPPOPolicy
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def rollout_buffer():
    return MultiAgentRolloutBuffer(
        buffer_size=20,
        num_agents=5,
        obs_dim=ENHANCED_OBS_DIM,
        act_dim=3,
        device="cpu",
    )


@pytest.fixture
def actor():
    return MAPPOActor(obs_dim=ENHANCED_OBS_DIM, act_dim=3)


@pytest.fixture
def policy():
    return MAPPOPolicy(
        obs_dim=ENHANCED_OBS_DIM,
        act_dim=3,
        critic_type="attention",
        device="cpu",
    )


@pytest.fixture
def trainer():
    config = MAPPOConfig(
        lr_actor=1e-3,
        lr_critic=1e-3,
        ppo_epochs=3,
        actor_mini_batch_size=32,
        critic_mini_batch_size=16,
        entropy_coef=0.01,
        use_popart=True,
    )
    return MAPPOTrainer(
        config=config,
        obs_dim=ENHANCED_OBS_DIM,
        num_agents=5,
        rollout_steps=15,
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════
# SUITE 1 — Multi-Agent Rollout Buffer
# ══════════════════════════════════════════════════════════════

class TestMultiAgentRolloutBuffer:
    """Validates trajectory storage, GAE calculation, and mini-batch generation."""

    def test_buffer_dimensions(self, rollout_buffer):
        assert rollout_buffer.obs.shape == (20, 5, ENHANCED_OBS_DIM)
        assert rollout_buffer.actions.shape == (20, 5, 3)
        assert rollout_buffer.rewards.shape == (20, 5)
        assert rollout_buffer.agent_states.shape == (20, 5, AGENT_FEATURE_DIM)
        assert rollout_buffer.env_states.shape == (20, ENV_FEATURE_DIM)
        assert rollout_buffer.values.shape == (21, 1)

    def test_insert_and_overflow(self, rollout_buffer):
        obs = np.ones((5, ENHANCED_OBS_DIM), dtype=np.float32)
        ag_states = np.zeros((5, AGENT_FEATURE_DIM), dtype=np.float32)
        env_state = np.zeros(ENV_FEATURE_DIM, dtype=np.float32)
        actions = np.zeros((5, 3), dtype=np.float32)
        log_probs = np.zeros(5, dtype=np.float32)
        rewards = np.ones(5, dtype=np.float32)
        dones = np.zeros(5, dtype=bool)

        for _ in range(20):
            rollout_buffer.insert(
                obs, ag_states, env_state, actions, log_probs, rewards, 1.0, dones
            )

        assert rollout_buffer.step == 20
        assert rollout_buffer.is_full

        # Next insert should raise overflow
        with pytest.raises(IndexError, match="Buffer overflow"):
            rollout_buffer.insert(
                obs, ag_states, env_state, actions, log_probs, rewards, 1.0, dones
            )

    def test_compute_returns_and_advantages(self, rollout_buffer):
        # Insert 10 steps
        for i in range(10):
            obs = np.random.randn(5, ENHANCED_OBS_DIM).astype(np.float32)
            ag = np.random.randn(5, AGENT_FEATURE_DIM).astype(np.float32)
            env = np.random.randn(ENV_FEATURE_DIM).astype(np.float32)
            act = np.random.uniform(-1, 1, (5, 3)).astype(np.float32)
            lp = np.random.randn(5).astype(np.float32)
            rew = np.random.randn(5).astype(np.float32)
            dones = np.zeros(5, dtype=bool)
            rollout_buffer.insert(obs, ag, env, act, lp, rew, float(i), dones)

        rollout_buffer.compute_returns_and_advantages(last_value=10.0)

        assert rollout_buffer.advantages[:10].shape == (10, 5)
        assert rollout_buffer.returns[:10].shape == (10, 5)
        assert rollout_buffer.joint_returns[:10].shape == (10, 1)

        # Normalized advantages should have zero mean
        norm_adv = rollout_buffer.normalized_advantages[:10]
        assert np.isclose(np.mean(norm_adv), 0.0, atol=1e-5)
        assert np.isclose(np.std(norm_adv), 1.0, atol=1e-3)

    def test_actor_mini_batch_generator(self, rollout_buffer):
        for _ in range(10):
            obs = np.ones((5, ENHANCED_OBS_DIM), dtype=np.float32)
            ag = np.zeros((5, AGENT_FEATURE_DIM), dtype=np.float32)
            env = np.zeros(ENV_FEATURE_DIM, dtype=np.float32)
            act = np.zeros((5, 3), dtype=np.float32)
            lp = np.zeros(5, dtype=np.float32)
            rew = np.ones(5, dtype=np.float32)
            dones = np.zeros(5, dtype=bool)
            rollout_buffer.insert(obs, ag, env, act, lp, rew, 1.0, dones)

        rollout_buffer.compute_returns_and_advantages(last_value=1.0)

        # Total samples = 10 steps * 5 agents = 50 samples
        gen = rollout_buffer.actor_mini_batch_generator(mini_batch_size=16, num_epochs=2)
        batches = list(gen)

        # 50 / 16 = 4 batches per epoch * 2 epochs = 8 batches
        assert len(batches) == 8
        for b in batches:
            assert "obs" in b
            assert "actions" in b
            assert "advantages" in b
            assert b["obs"].shape[-1] == ENHANCED_OBS_DIM

    def test_critic_mini_batch_generator(self, rollout_buffer):
        for _ in range(10):
            obs = np.ones((5, ENHANCED_OBS_DIM), dtype=np.float32)
            ag = np.zeros((5, AGENT_FEATURE_DIM), dtype=np.float32)
            env = np.zeros(ENV_FEATURE_DIM, dtype=np.float32)
            act = np.zeros((5, 3), dtype=np.float32)
            lp = np.zeros(5, dtype=np.float32)
            rew = np.ones(5, dtype=np.float32)
            dones = np.zeros(5, dtype=bool)
            rollout_buffer.insert(obs, ag, env, act, lp, rew, 1.0, dones)

        rollout_buffer.compute_returns_and_advantages(last_value=1.0)

        gen = rollout_buffer.critic_mini_batch_generator(mini_batch_size=5, num_epochs=1)
        batches = list(gen)
        assert len(batches) == 2  # 10 / 5 = 2 batches
        for b in batches:
            assert b["agent_states"].shape == (5, 5, AGENT_FEATURE_DIM)
            assert b["env_state"].shape == (5, ENV_FEATURE_DIM)
            assert b["target_returns"].shape == (5, 1)


# ══════════════════════════════════════════════════════════════
# SUITE 2 — Decentralized Actor & MAPPO Policy
# ══════════════════════════════════════════════════════════════

class TestMAPPOActorAndPolicy:
    """Validates actor Gaussian distribution, action sampling, and policy evaluation."""

    def test_actor_forward_and_shapes(self, actor):
        obs = torch.randn(8, ENHANCED_OBS_DIM)
        mean, std = actor(obs)

        assert mean.shape == (8, 3)
        assert std.shape == (8, 3)
        # Action means must be strictly bounded in [-1, 1] due to tanh
        assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)
        assert torch.all(std > 0.0)

    def test_action_sampling_modes(self, actor):
        obs = torch.randn(4, ENHANCED_OBS_DIM)

        # Deterministic mode
        mean, _ = actor(obs)
        det_act, det_lp = actor.get_action(obs, deterministic=True)
        assert torch.allclose(mean, det_act)

        # Stochastic mode
        torch.manual_seed(42)
        stoch_act, stoch_lp = actor.get_action(obs, deterministic=False)
        assert stoch_act.shape == (4, 3)
        assert torch.all(stoch_act >= -1.0) and torch.all(stoch_act <= 1.0)
        assert not torch.isnan(stoch_lp).any()

    def test_evaluate_actions_and_entropy(self, actor):
        obs = torch.randn(10, ENHANCED_OBS_DIM)
        actions, _ = actor.get_action(obs)

        log_probs, entropy = actor.evaluate_actions(obs, actions)

        assert log_probs.shape == (10,)
        assert entropy.shape == (10,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert torch.all(entropy > 0.0)  # Continuous Gaussian entropy is positive for std >= 1

    def test_policy_act_step(self, policy):
        N = 5
        obs_np = np.random.randn(N, ENHANCED_OBS_DIM).astype(np.float32)
        ag_np = np.random.randn(N, AGENT_FEATURE_DIM).astype(np.float32)
        env_np = np.random.randn(ENV_FEATURE_DIM).astype(np.float32)

        actions, log_probs, val = policy.act(obs_np, ag_np, env_np, deterministic=False)

        assert actions.shape == (N, 3)
        assert log_probs.shape == (N,)
        assert isinstance(val, float)
        assert not np.isnan(actions).any()
        assert not np.isnan(val)


# ══════════════════════════════════════════════════════════════
# SUITE 3 — MAPPO Loss Formulations
# ══════════════════════════════════════════════════════════════

class TestMAPPOLosses:
    """Validates PPO clipping, surrogate objective, and entropy bonus."""

    def test_ppo_clipped_surrogate_behavior(self):
        clip_param = 0.2
        advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])

        # Ratios: 1.0, 1.5 (clipped positive), 0.5 (clipped negative), 1.0
        ratios = torch.tensor([1.0, 1.5, 0.5, 1.0])

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        clipped_obj = torch.min(surr1, surr2)

        # For index 1 (adv=1.0, ratio=1.5): surr1=1.5, surr2=1.2 -> min=1.2
        assert torch.isclose(clipped_obj[1], torch.tensor(1.2))

        # For index 2 (adv=-1.0, ratio=0.5): surr1=-0.5, surr2=-0.8 -> min=-0.8
        assert torch.isclose(clipped_obj[2], torch.tensor(-0.8))

    def test_approx_kl_divergence(self):
        old_lp = torch.tensor([-1.2, -0.8, -2.1])
        new_lp = torch.tensor([-1.2, -0.8, -2.1])

        # Exactly zero when policies match
        kl_zero = 0.5 * torch.mean((old_lp - new_lp) ** 2).item()
        assert kl_zero == 0.0

        # Positive when policies differ
        diff_lp = torch.tensor([-1.0, -0.6, -1.9])
        kl_pos = 0.5 * torch.mean((old_lp - diff_lp) ** 2).item()
        assert kl_pos > 0.0


# ══════════════════════════════════════════════════════════════
# SUITE 4 — MAPPOTrainer Step & Checkpointing
# ══════════════════════════════════════════════════════════════

class TestMAPPOTrainer:
    """Validates complete rollout collection, training step, and model serialization."""

    def test_collect_rollout(self, trainer):
        env = CommSwarmEnv(num_drones=5, loss_prob=0.0)
        stats = trainer.collect_rollout(env, num_steps=15)

        assert trainer.buffer.step == 15
        assert "mean_reward" in stats
        assert "final_coverage" in stats
        assert "zone_a_coverage" in stats
        assert "zone_b_coverage" in stats
        assert not np.isnan(stats["mean_reward"])

    def test_train_step(self, trainer):
        env = CommSwarmEnv(num_drones=5, loss_prob=0.0)
        trainer.collect_rollout(env, num_steps=15)

        train_metrics = trainer.train_step()

        assert "policy_loss" in train_metrics
        assert "value_loss" in train_metrics
        assert "entropy" in train_metrics
        assert "approx_kl" in train_metrics
        assert "clip_fraction" in train_metrics
        assert "explained_variance" in train_metrics

        assert not np.isnan(train_metrics["policy_loss"])
        assert not np.isnan(train_metrics["value_loss"])

    def test_checkpoint_save_and_load(self, trainer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_mappo.pt")

            # Save checkpoint
            trainer.iteration = 42
            trainer.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            # Create fresh trainer and load checkpoint
            config = MAPPOConfig()
            new_trainer = MAPPOTrainer(config, obs_dim=ENHANCED_OBS_DIM, num_agents=5)
            new_trainer.load_checkpoint(ckpt_path)

            assert new_trainer.iteration == 42

            # Check actor weights match exactly
            for p1, p2 in zip(trainer.policy.actor.parameters(), new_trainer.policy.actor.parameters()):
                assert torch.equal(p1, p2)

            # Check critic weights match exactly
            for p1, p2 in zip(trainer.policy.critic.parameters(), new_trainer.policy.critic.parameters()):
                assert torch.equal(p1, p2)


# ══════════════════════════════════════════════════════════════
# SUITE 5 — Fleet Scalability & Multi-Agent Rollout Integration
# ══════════════════════════════════════════════════════════════

class TestFleetScalability:
    """Validates multi-agent training across fleet sizes and environment types."""

    def test_train_iteration_with_10_drones_comm(self):
        env = CommSwarmEnv(num_drones=10, loss_prob=0.0, formation_bonus=True)
        config = MAPPOConfig(ppo_epochs=2, actor_mini_batch_size=32, critic_mini_batch_size=16)
        trainer = MAPPOTrainer(config, obs_dim=ENHANCED_OBS_DIM, num_agents=10, rollout_steps=10)

        metrics = trainer.train_iteration(env, num_steps=10)

        assert metrics["iteration"] == 1
        assert metrics["final_coverage"] >= 0.0
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])

    def test_train_iteration_with_25_drones_comm(self):
        env = CommSwarmEnv(num_drones=25, loss_prob=0.0, formation_bonus=True)
        config = MAPPOConfig(ppo_epochs=1, actor_mini_batch_size=64, critic_mini_batch_size=16)
        trainer = MAPPOTrainer(config, obs_dim=ENHANCED_OBS_DIM, num_agents=25, rollout_steps=8)

        metrics = trainer.train_iteration(env, num_steps=8)

        assert metrics["iteration"] == 1
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])

    def test_train_iteration_with_base_swarm_env(self):
        """Validates compatibility with base 26-dim SwarmEnv."""
        env = SwarmEnv(num_drones=10)
        config = MAPPOConfig(ppo_epochs=1, actor_mini_batch_size=32, critic_mini_batch_size=16)
        trainer = MAPPOTrainer(config, obs_dim=OBS_DIM, num_agents=10, rollout_steps=10)

        metrics = trainer.train_iteration(env, num_steps=10)

        assert metrics["iteration"] == 1
        assert not np.isnan(metrics["policy_loss"])
