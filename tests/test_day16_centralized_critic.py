"""
SwarmRL -- Day 16: Centralized Critic Tests

Comprehensive test suite covering:
  Suite 1: Global State Extraction & Building (GlobalStateBuilder, SwarmEnv, CommSwarmEnv)
  Suite 2: Entity-Attention Centralized Critic (Shapes, Variable Fleet Size, Permutation Invariance, Gradients)
  Suite 3: MLP Centralized Critic Baseline (Feed-forward baseline, Backprop)
  Suite 4: Value Normalization & PopArt (Invertibility, Running Stats, Exact Parameter Adaptation)
  Suite 5: Multi-Agent GAE (Advantage shapes, TD-1 reduction, MC reduction, Terminal boundaries, Normalization)
  Suite 6: End-to-End CTDE Rollout & Training Optimization Loop
"""

import os
import sys
import pytest
import numpy as np
import torch
import torch.nn as nn

# ── path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.world_config import WORLD_CONFIG, GRID_CELLS_PER_AXIS, TOTAL_GRID_CELLS
from env.spaces import OBS_DIM
from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from training.centralized_critic import (
    GlobalStateBuilder,
    EntityAttentionCritic,
    MLPCentralizedCritic,
    ValueNorm,
    MultiAgentGAE,
    CentralizedCriticLoss,
    AGENT_FEATURE_DIM,
    ENV_FEATURE_DIM,
    MAX_DRONES,
    MAX_FLAT_STATE_DIM,
    WORLD_DIAG,
)


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def state_builder():
    return GlobalStateBuilder(max_drones=MAX_DRONES)


@pytest.fixture(scope="module")
def standard_env():
    env = SwarmEnv(num_drones=10)
    env.reset(seed=42)
    return env


@pytest.fixture(scope="module")
def comm_env():
    env = CommSwarmEnv(num_drones=10, loss_prob=0.0, formation_bonus=True)
    env.reset(seed=42)
    return env


# ══════════════════════════════════════════════════════════════
# SUITE 1 — Global State Extraction & Building
# ══════════════════════════════════════════════════════════════

class TestGlobalStateBuilder:
    """Validates global state extraction, normalization, and bounds."""

    def test_dimensions_and_constants(self):
        assert AGENT_FEATURE_DIM == 12
        assert ENV_FEATURE_DIM == 11
        assert MAX_DRONES == 50
        assert MAX_FLAT_STATE_DIM == 50 * 12 + 11
        assert np.isclose(WORLD_DIAG, 150.0)

    def test_extraction_from_swarm_env(self, standard_env):
        state = standard_env.get_global_state()

        assert "agent_states" in state
        assert "env_state" in state
        assert "flat_state" in state
        assert "padded_state" in state
        assert "agent_mask" in state
        assert "num_drones" in state
        assert "coverage" in state

        assert state["agent_states"].shape == (10, AGENT_FEATURE_DIM)
        assert state["env_state"].shape == (ENV_FEATURE_DIM,)
        assert state["flat_state"].shape == (10 * AGENT_FEATURE_DIM + ENV_FEATURE_DIM,)
        assert state["padded_state"].shape == (MAX_FLAT_STATE_DIM,)
        assert state["agent_mask"].shape == (MAX_DRONES,)
        assert state["num_drones"] == 10

    def test_extraction_from_comm_swarm_env(self, comm_env):
        state = comm_env.get_global_state()

        assert "agent_states" in state
        assert "env_state" in state
        assert "comm_stats" in state
        assert "mean_neighbours" in state["comm_stats"]
        assert "isolated_drones" in state["comm_stats"]
        assert "mean_separation" in state["comm_stats"]

        assert state["agent_states"].shape == (10, AGENT_FEATURE_DIM)
        assert state["env_state"].shape == (ENV_FEATURE_DIM,)

    def test_agent_feature_bounds(self, standard_env):
        state = standard_env.get_global_state()
        agents = state["agent_states"]

        # Position normalized [0, 1]
        assert np.all(agents[:, 0:3] >= 0.0) and np.all(agents[:, 0:3] <= 1.0)
        # Velocity normalized [-1, 1]
        assert np.all(agents[:, 3:6] >= -1.0) and np.all(agents[:, 3:6] <= 1.0)
        # Speed fraction [0, 1]
        assert np.all(agents[:, 6] >= 0.0) and np.all(agents[:, 6] <= 1.0)
        # Altitude fraction [0, 1]
        assert np.all(agents[:, 7] >= 0.0) and np.all(agents[:, 7] <= 1.0)
        # Nearest drone distance [0, 1]
        assert np.all(agents[:, 8] >= 0.0) and np.all(agents[:, 8] <= 1.0)
        # Nearest obstacle distance [0, 1]
        assert np.all(agents[:, 9] >= 0.0) and np.all(agents[:, 9] <= 1.0)
        # Cells explored fraction [0, 1]
        assert np.all(agents[:, 10] >= 0.0) and np.all(agents[:, 10] <= 1.0)
        # Alive flag in {0.0, 1.0}
        assert np.all(np.isin(agents[:, 11], [0.0, 1.0]))

    def test_env_feature_bounds(self, standard_env):
        state = standard_env.get_global_state()
        env_f = state["env_state"]

        assert env_f.shape == (11,)
        # All environment features must be in [0, 1]
        assert np.all(env_f >= 0.0) and np.all(env_f <= 1.0)
        assert not np.isnan(env_f).any()
        assert not np.isinf(env_f).any()

    @pytest.mark.parametrize("fleet_size", [5, 10, 25, 50])
    def test_variable_fleet_sizes(self, fleet_size):
        env = SwarmEnv(num_drones=fleet_size)
        env.reset(seed=123)
        state = env.get_global_state()

        assert state["agent_states"].shape == (fleet_size, AGENT_FEATURE_DIM)
        assert state["flat_state"].shape == (fleet_size * AGENT_FEATURE_DIM + ENV_FEATURE_DIM,)
        assert state["padded_state"].shape == (MAX_FLAT_STATE_DIM,)
        assert np.count_nonzero(state["agent_mask"]) == fleet_size

    def test_unpack_flat_state_roundtrip(self, standard_env):
        state = standard_env.get_global_state()
        flat_state = state["flat_state"]

        unpacked_agents, unpacked_env = GlobalStateBuilder.unpack_flat_state(flat_state)
        assert np.allclose(unpacked_agents, state["agent_states"])
        assert np.allclose(unpacked_env, state["env_state"])

    def test_unpack_flat_state_torch_batched(self):
        B, N = 4, 15
        dummy_flat = torch.randn(B, N * AGENT_FEATURE_DIM + ENV_FEATURE_DIM)
        agents, env_f = GlobalStateBuilder.unpack_flat_state(dummy_flat)

        assert agents.shape == (B, N, AGENT_FEATURE_DIM)
        assert env_f.shape == (B, ENV_FEATURE_DIM)

    def test_unpack_invalid_dimension_raises(self):
        invalid_flat = np.zeros(100)  # (100 - 11) = 89 not divisible by 12
        with pytest.raises(ValueError, match="Invalid flat_state dimension"):
            GlobalStateBuilder.unpack_flat_state(invalid_flat)


# ══════════════════════════════════════════════════════════════
# SUITE 2 — Entity-Attention Centralized Critic
# ══════════════════════════════════════════════════════════════

class TestEntityAttentionCritic:
    """Validates the permutation-invariant entity-attention centralized critic."""

    @pytest.fixture
    def critic(self):
        return EntityAttentionCritic(
            agent_feature_dim=AGENT_FEATURE_DIM,
            env_feature_dim=ENV_FEATURE_DIM,
            embed_dim=128,
            num_heads=4,
            hidden_dim=256,
        )

    def test_forward_unbatched(self, critic):
        N = 10
        agent_states = torch.rand(N, AGENT_FEATURE_DIM)
        env_state = torch.rand(ENV_FEATURE_DIM)

        v_joint = critic(agent_states, env_state)
        assert v_joint.shape == (1,)
        assert not torch.isnan(v_joint).any()

    def test_forward_batched(self, critic):
        B, N = 8, 12
        agent_states = torch.rand(B, N, AGENT_FEATURE_DIM)
        env_state = torch.rand(B, ENV_FEATURE_DIM)

        v_joint = critic(agent_states, env_state)
        assert v_joint.shape == (B, 1)
        assert not torch.isnan(v_joint).any()

    def test_forward_with_flat_state(self, critic):
        B, N = 4, 10
        flat_dim = N * AGENT_FEATURE_DIM + ENV_FEATURE_DIM
        flat_state = torch.rand(B, flat_dim)

        v_joint = critic(flat_state)
        assert v_joint.shape == (B, 1)

    @pytest.mark.parametrize("fleet_size", [5, 10, 25, 50])
    def test_variable_fleet_size_compatibility(self, critic, fleet_size):
        """The same model instance must evaluate fleets of 5, 10, 25, 50 drones."""
        agent_states = torch.rand(2, fleet_size, AGENT_FEATURE_DIM)
        env_state = torch.rand(2, ENV_FEATURE_DIM)

        v_joint = critic(agent_states, env_state)
        assert v_joint.shape == (2, 1)

    def test_individual_value_baselines(self, critic):
        B, N = 4, 15
        agent_states = torch.rand(B, N, AGENT_FEATURE_DIM)
        env_state = torch.rand(B, ENV_FEATURE_DIM)

        v_joint, v_agents = critic(agent_states, env_state, return_individual=True)
        assert v_joint.shape == (B, 1)
        assert v_agents.shape == (B, N)
        assert not torch.isnan(v_agents).any()

    def test_permutation_invariance(self, critic):
        """
        Swapping drone row 0 and row 1 must yield the exact same joint value V(s),
        and swapped individual baselines V_0(s) <-> V_1(s).
        """
        torch.manual_seed(42)
        N = 10
        agent_states = torch.rand(1, N, AGENT_FEATURE_DIM)
        env_state = torch.rand(1, ENV_FEATURE_DIM)

        # Baseline evaluation
        v_orig, v_indiv_orig = critic(agent_states, env_state, return_individual=True)

        # Permute agents: swap index 0 and index 3
        perm = list(range(N))
        perm[0], perm[3] = perm[3], perm[0]
        permuted_agents = agent_states[:, perm, :]

        # Permuted evaluation
        v_perm, v_indiv_perm = critic(permuted_agents, env_state, return_individual=True)

        # Joint value V(s) must be strictly invariant
        diff_joint = torch.abs(v_orig - v_perm).item()
        assert diff_joint < 1e-5, f"Joint value violated permutation invariance: diff={diff_joint}"

        # Individual values must be equivariant (permuted accordingly)
        assert torch.allclose(v_indiv_orig[:, 0], v_indiv_perm[:, 3], atol=1e-5)
        assert torch.allclose(v_indiv_orig[:, 3], v_indiv_perm[:, 0], atol=1e-5)

    def test_agent_masking(self, critic):
        B, N = 2, 20
        agent_states = torch.rand(B, N, AGENT_FEATURE_DIM)
        env_state = torch.rand(B, ENV_FEATURE_DIM)

        # Mask out last 5 drones
        agent_mask = torch.ones(B, N, dtype=torch.bool)
        agent_mask[:, 15:] = False

        v_masked = critic(agent_states, env_state, agent_mask=agent_mask)
        assert v_masked.shape == (B, 1)
        assert not torch.isnan(v_masked).any()

    def test_gradient_flow(self, critic):
        agent_states = torch.rand(4, 10, AGENT_FEATURE_DIM, requires_grad=True)
        env_state = torch.rand(4, ENV_FEATURE_DIM, requires_grad=True)

        v_joint, v_agents = critic(agent_states, env_state, return_individual=True)
        loss = v_joint.sum() + v_agents.sum()
        loss.backward()

        assert agent_states.grad is not None
        assert not torch.isnan(agent_states.grad).any()
        assert env_state.grad is not None
        assert not torch.isnan(env_state.grad).any()

        # Check model weights have gradients
        for name, param in critic.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Param {name} missing gradient"


# ══════════════════════════════════════════════════════════════
# SUITE 3 — MLP Centralized Critic Baseline
# ══════════════════════════════════════════════════════════════

class TestMLPCentralizedCritic:
    """Validates the standard MLP centralized critic baseline."""

    @pytest.fixture
    def mlp_critic(self):
        return MLPCentralizedCritic(input_dim=MAX_FLAT_STATE_DIM)

    def test_forward_unbatched(self, mlp_critic):
        flat_state = torch.rand(MAX_FLAT_STATE_DIM)
        val = mlp_critic(flat_state)
        assert val.shape == (1,)
        assert not torch.isnan(val).any()

    def test_forward_batched(self, mlp_critic):
        B = 8
        flat_batch = torch.rand(B, MAX_FLAT_STATE_DIM)
        val = mlp_critic(flat_batch)
        assert val.shape == (B, 1)

    def test_backprop(self, mlp_critic):
        flat_batch = torch.rand(4, MAX_FLAT_STATE_DIM, requires_grad=True)
        val = mlp_critic(flat_batch)
        loss = val.sum()
        loss.backward()
        assert flat_batch.grad is not None
        for p in mlp_critic.parameters():
            assert p.grad is not None


# ══════════════════════════════════════════════════════════════
# SUITE 4 — Value Normalization & PopArt
# ══════════════════════════════════════════════════════════════

class TestValueNormPopArt:
    """Validates adaptive value target normalization and PopArt parameter preservation."""

    @pytest.fixture
    def val_norm(self):
        return ValueNorm(beta=0.99, epsilon=1e-5)

    def test_initial_state(self, val_norm):
        assert val_norm.running_mean.item() == 0.0
        assert val_norm.running_var.item() == 1.0
        assert val_norm.running_std.item() == 1.0
        assert val_norm.count.item() == 0

    def test_normalization_denormalization_invertibility(self, val_norm):
        # Set arbitrary running stats
        val_norm.running_mean.copy_(torch.tensor([25.0]))
        val_norm.running_var.copy_(torch.tensor([16.0]))
        val_norm.running_std.copy_(torch.tensor([4.0]))

        targets = torch.tensor([10.0, 25.0, 33.0, 50.0])
        normed = val_norm.normalize(targets)
        denormed = val_norm.denormalize(normed)

        assert torch.allclose(targets, denormed, atol=1e-5)

    def test_numpy_support(self, val_norm):
        val_norm.running_mean.copy_(torch.tensor([10.0]))
        val_norm.running_std.copy_(torch.tensor([2.0]))

        targets_np = np.array([6.0, 10.0, 14.0], dtype=np.float32)
        normed_np = val_norm.normalize(targets_np)
        denormed_np = val_norm.denormalize(normed_np)

        assert np.allclose(normed_np, np.array([-2.0, 0.0, 2.0]))
        assert np.allclose(denormed_np, targets_np)

    def test_running_statistics_update(self, val_norm):
        # Update with high return targets (e.g. +100 bonus)
        high_targets = torch.full((100,), 100.0)
        val_norm.update(high_targets)

        assert val_norm.running_mean.item() > 0.0
        assert val_norm.count.item() == 100

    def test_popart_exact_output_preservation(self, val_norm):
        """
        PopArt guarantee: updating stats and adjusting layer weights/biases
        must leave unnormalized predictions unchanged!
        """
        torch.manual_seed(99)
        val_norm = ValueNorm(beta=0.90, epsilon=1e-5)

        # Simulated last linear layer
        layer = nn.Linear(32, 1)
        x = torch.randn(10, 32)

        # Baseline predictions before update
        y_norm_old = layer(x)
        y_unnorm_old = val_norm.denormalize(y_norm_old)

        # New targets with large shift
        new_targets = torch.randn(50) * 15.0 + 80.0

        # Perform PopArt update
        val_norm.update_popart(layer, new_targets)

        # Predictions after update
        y_norm_new = layer(x)
        y_unnorm_new = val_norm.denormalize(y_norm_new)

        # Verify discrepancy is within numerical tolerance
        max_diff = torch.max(torch.abs(y_unnorm_old - y_unnorm_new)).item()
        assert max_diff < 1e-5, f"PopArt output shift was {max_diff}"


# ══════════════════════════════════════════════════════════════
# SUITE 5 — Multi-Agent Generalized Advantage Estimation (GAE)
# ══════════════════════════════════════════════════════════════

class TestMultiAgentGAE:
    """Validates Generalized Advantage Estimation computation."""

    def test_gae_shapes(self):
        T, N = 20, 10
        rewards = np.random.randn(T, N).astype(np.float32)
        values = np.random.randn(T + 1, N).astype(np.float32)
        dones = np.zeros((T, N), dtype=bool)

        res = MultiAgentGAE.compute(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        assert res["advantages"].shape == (T, N)
        assert res["returns"].shape == (T, N)
        assert res["normalized_advantages"].shape == (T, N)

    def test_gae_zero_future_td1_reduction(self):
        """When gae_lambda = 0, GAE reduces to 1-step TD error."""
        T, N = 5, 2
        rewards = np.array([[1.0, 2.0], [0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
        values = np.zeros((T + 1, N), dtype=np.float32)
        values[1] = [10.0, 10.0]
        dones = np.zeros((T, N), dtype=bool)

        gamma = 0.9
        res = MultiAgentGAE.compute(rewards, values, dones, gamma=gamma, gae_lambda=0.0, normalize_adv=False)

        # At t=0: delta = r_0 + gamma * v_1 - v_0 = [1.0, 2.0] + 0.9 * [10.0, 10.0] - 0 = [10.0, 11.0]
        expected_adv_0 = rewards[0] + gamma * values[1] - values[0]
        assert np.allclose(res["advantages"][0], expected_adv_0)

    def test_gae_terminal_boundary_masking(self):
        """When done=True at step t, future values must not propagate."""
        T, N = 4, 1
        rewards = np.ones((T, N), dtype=np.float32)
        values = np.ones((T + 1, N), dtype=np.float32) * 5.0
        dones = np.zeros((T, N), dtype=bool)
        dones[1] = True  # Terminal at step 1

        res = MultiAgentGAE.compute(rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_adv=False)

        # At t=1: delta_1 = r_1 + gamma * 0 - v_1 = 1.0 - 5.0 = -4.0
        assert np.isclose(res["advantages"][1, 0], -4.0)

    def test_gae_torch_tensor_support(self):
        T, N = 10, 4
        rewards = torch.randn(T, N)
        values = torch.randn(T + 1, N)
        dones = torch.zeros(T, N, dtype=torch.bool)

        res = MultiAgentGAE.compute(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        assert isinstance(res["advantages"], torch.Tensor)
        assert isinstance(res["returns"], torch.Tensor)
        assert res["advantages"].shape == (T, N)

    def test_advantage_normalization(self):
        T, N = 50, 10
        rewards = np.random.randn(T, N).astype(np.float32)
        values = np.random.randn(T + 1, N).astype(np.float32)
        dones = np.zeros((T, N), dtype=bool)

        res = MultiAgentGAE.compute(rewards, values, dones, normalize_adv=True)
        norm_adv = res["normalized_advantages"]

        assert np.isclose(np.mean(norm_adv), 0.0, atol=1e-5)
        assert np.isclose(np.std(norm_adv), 1.0, atol=1e-3)


# ══════════════════════════════════════════════════════════════
# SUITE 6 — End-to-End CTDE Rollout & Training Optimization
# ══════════════════════════════════════════════════════════════

class TestCentralizedCriticEndToEnd:
    """Validates multi-step CTDE rollout with critic updates."""

    def test_critic_loss_functions(self):
        v_pred = torch.tensor([10.0, 20.0, 30.0])
        returns = torch.tensor([12.0, 18.0, 35.0])

        # MSE
        loss_mse = CentralizedCriticLoss.compute_loss(v_pred, returns)
        assert loss_mse.item() > 0.0

        # Clipped Loss
        v_old = torch.tensor([10.5, 19.5, 29.5])
        loss_clipped = CentralizedCriticLoss.compute_loss(v_pred, returns, v_old=v_old, clip_param=0.2)
        assert loss_clipped.item() > 0.0

        # Huber Loss
        loss_huber = CentralizedCriticLoss.compute_loss(v_pred, returns, use_huber=True)
        assert loss_huber.item() > 0.0

    def test_ctde_rollout_and_critic_training_step(self):
        """
        Executes a 10-step simulation rollout in SwarmEnv,
        extracts global states, computes GAE returns, and performs a gradient step.
        """
        num_drones = 10
        env = SwarmEnv(num_drones=num_drones)
        obs, _ = env.reset(seed=42)

        critic = EntityAttentionCritic(
            agent_feature_dim=AGENT_FEATURE_DIM,
            env_feature_dim=ENV_FEATURE_DIM,
            embed_dim=64,
            num_heads=2,
            hidden_dim=128,
        )
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        val_norm = ValueNorm()

        rollout_agent_states = []
        rollout_env_states = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        # 10 simulation steps
        for step_idx in range(10):
            global_state = env.get_global_state()
            ag_t = torch.as_tensor(global_state["agent_states"], dtype=torch.float32)
            env_t = torch.as_tensor(global_state["env_state"], dtype=torch.float32)

            with torch.no_grad():
                v = critic(ag_t, env_t)  # shape (1,)

            rollout_agent_states.append(ag_t)
            rollout_env_states.append(env_t)
            rollout_values.append(v.item())

            # Step environment with dummy actions
            actions = {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}
            obs, rewards, term, trunc, _ = env.step(actions)

            mean_reward = float(np.mean(list(rewards.values())))
            rollout_rewards.append(mean_reward)
            rollout_dones.append(False)

        # Bootstrap value at step T
        final_state = env.get_global_state()
        ag_final = torch.as_tensor(final_state["agent_states"], dtype=torch.float32)
        env_final = torch.as_tensor(final_state["env_state"], dtype=torch.float32)
        with torch.no_grad():
            v_final = critic(ag_final, env_final).item()
        rollout_values.append(v_final)

        # Compute GAE
        rewards_arr = np.array(rollout_rewards)[:, None]  # (10, 1)
        values_arr = np.array(rollout_values)[:, None]    # (11, 1)
        dones_arr = np.array(rollout_dones)[:, None]      # (10, 1)

        gae_res = MultiAgentGAE.compute(rewards_arr, values_arr, dones_arr)
        target_returns = torch.as_tensor(gae_res["returns"], dtype=torch.float32)

        # Update value norm with target returns
        val_norm.update(target_returns)
        norm_targets = val_norm.normalize(target_returns)

        # Stack batch
        batch_ag = torch.stack(rollout_agent_states)  # (10, 10, 12)
        batch_env = torch.stack(rollout_env_states)   # (10, 11)

        # Train critic for optimization iterations
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        initial_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            v_pred = critic(batch_ag, batch_env)
            loss = CentralizedCriticLoss.compute_loss(v_pred, norm_targets)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        # Re-evaluate
        with torch.no_grad():
            v_pred_after = critic(batch_ag, batch_env)
            final_loss = CentralizedCriticLoss.compute_loss(v_pred_after, norm_targets).item()

        assert not np.isnan(initial_loss)
        assert not np.isnan(final_loss)
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
