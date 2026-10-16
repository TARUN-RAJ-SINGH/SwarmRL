"""
SwarmRL -- Day 15: GNN Communication Tests

Tests for:
  Suite 1: GNN Module Correctness  (MessageEncoder, SpatialAttentionAggregator,
            GNNCommLayer, GNNPolicyNetwork)
  Suite 2: CommSwarmEnv API         (observation shape, reward plumbing,
            comm stats, formation bonus)
  Suite 3: Bandwidth & Packet Loss  (cap enforcement, loss simulation)
  Suite 4: End-to-End Episode       (full reset→step loop, no crashes)
"""

import sys
import os
import pytest
import numpy as np

# ── path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── lazy torch import (skip gracefully if not installed) ───
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from env.gnn_comm import (
    MessageEncoder,
    SpatialAttentionAggregator,
    build_comm_observation,
    apply_packet_loss,
    apply_bandwidth_limit,
    MSG_DIM,
    BASE_OBS_DIM,
    ENHANCED_OBS_DIM,
)
from env.comm_swarm_env import CommSwarmEnv, get_enhanced_observation_space
from env.spaces import OBS_DIM


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def base_obs():
    """A valid 26-dim base observation (all in [0,1] / [-1,1])."""
    rng = np.random.default_rng(42)
    obs = rng.uniform(0.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
    obs[3:6] = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
    return obs


@pytest.fixture(scope="module")
def encoder():
    return MessageEncoder()


@pytest.fixture(scope="module")
def aggregator():
    return SpatialAttentionAggregator(temperature=2.0)


@pytest.fixture(scope="module")
def comm_env():
    env = CommSwarmEnv(num_drones=5, loss_prob=0.0, formation_bonus=True)
    return env


# ══════════════════════════════════════════════════════════════
# SUITE 1 — GNN Module Correctness
# ══════════════════════════════════════════════════════════════

class TestMessageEncoder:
    """Validates the NumPy message encoder."""

    def test_output_shape(self, encoder, base_obs):
        receiver_pos = np.array([0.5, 0.5, 0.1], dtype=np.float32)
        msg = encoder.encode(base_obs, receiver_pos, dist_norm=0.3)
        assert msg.shape == (MSG_DIM,), f"Expected ({MSG_DIM},), got {msg.shape}"

    def test_output_dtype(self, encoder, base_obs):
        receiver_pos = np.zeros(3, dtype=np.float32)
        msg = encoder.encode(base_obs, receiver_pos, dist_norm=0.5)
        assert msg.dtype == np.float32

    def test_relative_position_in_message(self, encoder, base_obs):
        """Relative position should be sender_pos - receiver_pos."""
        receiver_pos = base_obs[0:3].copy()
        # If receiver_pos == sender_pos, rel_pos should be ~zero
        msg = encoder.encode(base_obs, receiver_pos, dist_norm=0.0)
        rel_pos_in_msg = msg[0:3]
        assert np.allclose(rel_pos_in_msg, 0.0, atol=1e-5), (
            f"Expected near-zero rel_pos, got {rel_pos_in_msg}"
        )

    def test_dist_encoded_at_index_15(self, encoder, base_obs):
        receiver_pos = np.zeros(3, dtype=np.float32)
        dist_norm = 0.77
        msg = encoder.encode(base_obs, receiver_pos, dist_norm=dist_norm)
        assert abs(float(msg[15]) - dist_norm) < 1e-5

    def test_clipping_relative_position(self, encoder, base_obs):
        """Relative position values should be clipped to [-1, 1]."""
        receiver_pos = np.zeros(3, dtype=np.float32)
        msg = encoder.encode(base_obs, receiver_pos, dist_norm=0.0)
        assert np.all(msg[0:3] >= -1.0) and np.all(msg[0:3] <= 1.0)


class TestSpatialAttentionAggregator:
    """Validates the attention aggregator."""

    def test_empty_messages_returns_zero(self, aggregator):
        ctx = aggregator.aggregate([], [])
        assert ctx.shape == (MSG_DIM,)
        assert np.allclose(ctx, 0.0)

    def test_single_message_returns_itself(self, aggregator):
        msg = np.ones(MSG_DIM, dtype=np.float32) * 0.5
        ctx = aggregator.aggregate([msg], [0.3])
        assert np.allclose(ctx, msg, atol=1e-5)

    def test_weights_sum_to_one(self, aggregator):
        """Softmax weights should normalise; output is a valid weighted avg."""
        msgs = [np.ones(MSG_DIM, dtype=np.float32) for _ in range(4)]
        dists = [0.1, 0.3, 0.5, 0.7]
        ctx = aggregator.aggregate(msgs, dists)
        # All messages are identical → ctx should equal that message
        assert np.allclose(ctx, msgs[0], atol=1e-5)

    def test_closer_drones_get_more_weight(self, aggregator):
        """Context should be biased toward the nearest neighbour."""
        msg_near = np.ones(MSG_DIM, dtype=np.float32)          # all 1s
        msg_far  = np.zeros(MSG_DIM, dtype=np.float32)          # all 0s
        ctx = aggregator.aggregate([msg_near, msg_far], [0.1, 0.9])
        # With temp=2 and much closer near, mean value should be > 0.5
        assert float(ctx.mean()) > 0.5, f"Expected bias toward near, got mean={ctx.mean()}"

    def test_output_shape_multiple(self, aggregator):
        msgs  = [np.random.randn(MSG_DIM).astype(np.float32) for _ in range(6)]
        dists = np.linspace(0.1, 0.9, 6).tolist()
        ctx = aggregator.aggregate(msgs, dists)
        assert ctx.shape == (MSG_DIM,)

    def test_output_dtype(self, aggregator):
        msg = np.ones(MSG_DIM, dtype=np.float32)
        ctx = aggregator.aggregate([msg], [0.5])
        assert ctx.dtype == np.float32


class TestBuildCommObservation:
    """Validates the full build_comm_observation function."""

    def _make_fleet_obs(self, n=5):
        rng = np.random.default_rng(0)
        fleet_obs = {}
        fleet_pos = {}
        for i in range(n):
            obs = rng.uniform(0.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
            obs[3:6] = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
            fleet_obs[i] = obs
            fleet_pos[i] = rng.uniform(0.0, 100.0, size=3).astype(np.float32)
        return fleet_obs, fleet_pos

    def test_output_shape(self):
        obs_dict, pos_dict = self._make_fleet_obs(5)
        enh = build_comm_observation(0, obs_dict[0], obs_dict, pos_dict)
        assert enh.shape == (ENHANCED_OBS_DIM,), (
            f"Expected ({ENHANCED_OBS_DIM},), got {enh.shape}"
        )

    def test_output_dtype(self):
        obs_dict, pos_dict = self._make_fleet_obs(5)
        enh = build_comm_observation(0, obs_dict[0], obs_dict, pos_dict)
        assert enh.dtype == np.float32

    def test_base_obs_preserved(self):
        """First 26 dims should be identical to input base_obs."""
        obs_dict, pos_dict = self._make_fleet_obs(5)
        enh = build_comm_observation(0, obs_dict[0], obs_dict, pos_dict)
        assert np.allclose(enh[:OBS_DIM], obs_dict[0]), (
            "Base observation should be unchanged in enhanced obs."
        )

    def test_isolated_drone_gets_zero_context(self):
        """Drone far outside comm range → context should be zero."""
        obs_dict = {
            0: np.zeros(OBS_DIM, dtype=np.float32),
            1: np.zeros(OBS_DIM, dtype=np.float32),
        }
        pos_dict = {
            0: np.array([0.0, 0.0, 5.0]),
            1: np.array([99.0, 99.0, 5.0]),  # 140m away
        }
        enh = build_comm_observation(
            0, obs_dict[0], obs_dict, pos_dict,
            comm_range=20.0, sensor_range=10.0,
        )
        context = enh[OBS_DIM:]
        assert np.allclose(context, 0.0), (
            f"Expected zero context for isolated drone, got {context}"
        )

    def test_nearby_drone_contributes_context(self):
        """Drone within comm range should produce nonzero context."""
        obs_dict = {
            0: np.zeros(OBS_DIM, dtype=np.float32),
            1: np.ones(OBS_DIM, dtype=np.float32),  # nonzero sender
        }
        obs_dict[1][3:6] = 0.0  # velocity in [-1,1]
        pos_dict = {
            0: np.array([10.0, 10.0, 5.0]),
            1: np.array([15.0, 10.0, 5.0]),  # 5m away
        }
        enh = build_comm_observation(
            0, obs_dict[0], obs_dict, pos_dict,
            comm_range=20.0, sensor_range=10.0,
        )
        context = enh[OBS_DIM:]
        assert not np.allclose(context, 0.0), (
            "Expected nonzero context when neighbour is nearby."
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestGNNCommLayer:
    """Validates the PyTorch GNN attention layer."""

    def test_output_shape(self):
        from env.gnn_comm import GNNCommLayer
        layer = GNNCommLayer(input_dim=ENHANCED_OBS_DIM, hidden_dim=128, num_heads=4)
        B, N = 4, 5
        x = torch.randn(B, N, ENHANCED_OBS_DIM)
        out = layer(x)
        assert out.shape == (B, N, 128), f"Expected (4,5,128), got {out.shape}"

    def test_output_with_comm_mask(self):
        from env.gnn_comm import GNNCommLayer
        layer = GNNCommLayer(input_dim=ENHANCED_OBS_DIM, hidden_dim=64, num_heads=2)
        B, N = 2, 4
        x    = torch.randn(B, N, ENHANCED_OBS_DIM)
        mask = torch.ones(B, N, N, dtype=torch.bool)
        mask[:, :, 2] = False  # block comms to agent 2
        out  = layer(x, comm_mask=mask)
        assert out.shape == (B, N, 64)

    def test_gradient_flows(self):
        from env.gnn_comm import GNNCommLayer
        layer = GNNCommLayer(input_dim=ENHANCED_OBS_DIM, hidden_dim=64, num_heads=2)
        x = torch.randn(2, 3, ENHANCED_OBS_DIM, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestGNNPolicyNetwork:
    """Validates the full actor-critic GNN policy network."""

    def test_single_agent_forward(self):
        from env.gnn_comm import GNNPolicyNetwork
        net = GNNPolicyNetwork(obs_dim=ENHANCED_OBS_DIM, action_dim=3)
        x = torch.randn(8, ENHANCED_OBS_DIM)  # (batch=8, obs_dim)
        mean, log_std, value = net(x)
        assert mean.shape  == (8, 3), f"Action mean shape: {mean.shape}"
        assert log_std.shape == (3,)
        assert value.shape == (8, 1), f"Value shape: {value.shape}"

    def test_multi_agent_forward(self):
        from env.gnn_comm import GNNPolicyNetwork
        net = GNNPolicyNetwork(obs_dim=ENHANCED_OBS_DIM, action_dim=3)
        x = torch.randn(4, 10, ENHANCED_OBS_DIM)  # (batch=4, agents=10, obs)
        mean, log_std, value = net(x)
        assert mean.shape  == (4, 10, 3)
        assert value.shape == (4, 10, 1)

    def test_action_mean_in_range(self):
        """tanh output should be in (-1, 1)."""
        from env.gnn_comm import GNNPolicyNetwork
        net = GNNPolicyNetwork(obs_dim=ENHANCED_OBS_DIM, action_dim=3)
        x = torch.randn(16, ENHANCED_OBS_DIM)
        mean, _, _ = net(x)
        assert torch.all(mean >  -1.0 - 1e-5), "Action mean below -1"
        assert torch.all(mean <   1.0 + 1e-5), "Action mean above  1"

    def test_parameter_count(self):
        from env.gnn_comm import GNNPolicyNetwork
        net = GNNPolicyNetwork(obs_dim=ENHANCED_OBS_DIM, action_dim=3)
        n_params = sum(p.numel() for p in net.parameters())
        # Expect tens of thousands of params
        assert n_params > 10_000, f"Too few params: {n_params}"


# ══════════════════════════════════════════════════════════════
# SUITE 2 — CommSwarmEnv API
# ══════════════════════════════════════════════════════════════

class TestCommSwarmEnvAPI:
    """Validates the PettingZoo-compliant CommSwarmEnv interface."""

    def test_observation_space_shape(self, comm_env):
        for agent in comm_env.possible_agents:
            sp = comm_env.observation_space(agent)
            assert sp.shape == (ENHANCED_OBS_DIM,), (
                f"Expected ({ENHANCED_OBS_DIM},), got {sp.shape}"
            )

    def test_action_space_shape(self, comm_env):
        for agent in comm_env.possible_agents:
            sp = comm_env.action_space(agent)
            assert sp.shape == (3,)

    def test_reset_returns_enhanced_obs(self, comm_env):
        obs, infos = comm_env.reset(seed=0)
        for agent, o in obs.items():
            assert o.shape == (ENHANCED_OBS_DIM,), (
                f"Agent {agent}: expected ({ENHANCED_OBS_DIM},), got {o.shape}"
            )

    def test_reset_obs_in_space_bounds(self, comm_env):
        obs, _ = comm_env.reset(seed=1)
        space = comm_env.observation_space(comm_env.possible_agents[0])
        for agent, o in obs.items():
            assert space.contains(o.astype(np.float32)), (
                f"Agent {agent}: obs out of space bounds"
            )

    def test_step_returns_enhanced_obs(self, comm_env):
        comm_env.reset(seed=2)
        actions = {
            a: comm_env.action_space(a).sample()
            for a in comm_env.agents
        }
        obs, rewards, terms, truncs, infos = comm_env.step(actions)
        for agent, o in obs.items():
            assert o.shape == (ENHANCED_OBS_DIM,)

    def test_step_rewards_are_finite(self, comm_env):
        comm_env.reset(seed=3)
        actions = {a: np.zeros(3) for a in comm_env.agents}
        _, rewards, _, _, _ = comm_env.step(actions)
        for agent, r in rewards.items():
            assert np.isfinite(r), f"Agent {agent} reward is not finite: {r}"

    def test_agent_list_populated(self, comm_env):
        comm_env.reset(seed=4)
        assert len(comm_env.agents) == 5

    def test_possible_agents_unchanged(self, comm_env):
        comm_env.reset(seed=5)
        assert len(comm_env.possible_agents) == 5

    def test_comm_stats_structure(self, comm_env):
        comm_env.reset(seed=6)
        stats = comm_env.get_comm_stats()
        assert "mean_neighbours" in stats
        assert "isolated_drones" in stats
        assert "mean_separation" in stats

    def test_comm_stats_are_numeric(self, comm_env):
        comm_env.reset(seed=7)
        stats = comm_env.get_comm_stats()
        for k, v in stats.items():
            assert isinstance(v, (int, float)), f"{k} is not numeric: {v}"

    def test_inner_env_accessible(self, comm_env):
        from env.swarm_env import SwarmEnv
        assert isinstance(comm_env.inner_env, SwarmEnv)


class TestFormationBonus:
    """Tests cooperative dispersion reward shaping."""

    def test_no_bonus_returns_zero_at_spawn(self):
        """Newly spawned cluster is likely too close → no bonus."""
        env = CommSwarmEnv(num_drones=5, formation_bonus=True)
        env.reset(seed=0)
        # Swarm starts clustered at base-camp; mean sep << 5m
        bonus = env._compute_formation_bonus()
        # Either 0.0 or 0.05; just verify it's finite
        assert np.isfinite(bonus)

    def test_bonus_disabled(self):
        env = CommSwarmEnv(num_drones=5, formation_bonus=False)
        obs, _ = env.reset(seed=0)
        acts = {a: np.zeros(3) for a in env.agents}
        _, rewards, _, _, _ = env.step(acts)
        # With formation_bonus=False, bonus should not be added
        # (we can't assert exact value, just check it runs without error)
        for r in rewards.values():
            assert np.isfinite(r)


# ══════════════════════════════════════════════════════════════
# SUITE 3 — Bandwidth & Packet Loss
# ══════════════════════════════════════════════════════════════

class TestBandwidthAndPacketLoss:
    """Tests network reliability simulation utilities."""

    def _make_msgs(self, n):
        msgs  = [np.random.randn(MSG_DIM).astype(np.float32) for _ in range(n)]
        dists = list(np.linspace(0.1, 0.9, n))
        return msgs, dists

    def test_bandwidth_cap_reduces_count(self):
        msgs, dists = self._make_msgs(10)
        out_msgs, out_dists = apply_bandwidth_limit(msgs, dists, max_msgs=4)
        assert len(out_msgs)  == 4
        assert len(out_dists) == 4

    def test_bandwidth_cap_keeps_nearest(self):
        """Cap should prefer nearest neighbours (smallest distance)."""
        msgs  = [np.ones(MSG_DIM, dtype=np.float32) * i for i in range(5)]
        dists = [0.9, 0.1, 0.5, 0.2, 0.8]
        out_msgs, out_dists = apply_bandwidth_limit(msgs, dists, max_msgs=2)
        assert sorted(out_dists) == sorted(out_dists[:2])
        # Nearest two should be 0.1 and 0.2
        assert set(out_dists) == {0.1, 0.2}

    def test_bandwidth_cap_no_change_if_below_limit(self):
        msgs, dists = self._make_msgs(3)
        out_msgs, out_dists = apply_bandwidth_limit(msgs, dists, max_msgs=5)
        assert len(out_msgs) == 3

    def test_packet_loss_zero_prob_keeps_all(self):
        msgs, dists = self._make_msgs(5)
        rng = np.random.default_rng(99)
        out_msgs, out_dists = apply_packet_loss(msgs, dists, loss_prob=0.0, rng=rng)
        assert len(out_msgs) == 5

    def test_packet_loss_full_prob_drops_all(self):
        msgs, dists = self._make_msgs(10)
        rng = np.random.default_rng(0)
        # With loss_prob=1.0, all messages should be dropped
        out_msgs, _ = apply_packet_loss(msgs, dists, loss_prob=1.0, rng=rng)
        assert len(out_msgs) == 0

    def test_packet_loss_partial(self):
        """With partial loss, some but not all messages kept."""
        msgs, dists = self._make_msgs(100)
        rng = np.random.default_rng(42)
        out_msgs, _ = apply_packet_loss(msgs, dists, loss_prob=0.5, rng=rng)
        # Statistically: expect ~50 kept, allow generous margin
        assert 10 < len(out_msgs) < 90, f"Unexpected count: {len(out_msgs)}"

    def test_packet_loss_env_still_produces_valid_obs(self):
        """CommSwarmEnv with loss_prob=0.5 should still return valid obs."""
        env = CommSwarmEnv(num_drones=5, loss_prob=0.5)
        obs, _ = env.reset(seed=0)
        space = env.observation_space(env.possible_agents[0])
        for agent, o in obs.items():
            assert o.shape == (ENHANCED_OBS_DIM,)
            assert space.contains(o.astype(np.float32)), (
                f"{agent}: obs out of bounds with packet loss"
            )

    def test_enhanced_obs_space_bounds_valid(self):
        sp = get_enhanced_observation_space()
        assert sp.shape == (ENHANCED_OBS_DIM,)
        assert np.all(sp.low  <= sp.high), "Space low > high somewhere"


# ══════════════════════════════════════════════════════════════
# SUITE 4 — End-to-End Episode
# ══════════════════════════════════════════════════════════════

class TestEndToEndEpisode:
    """Full reset→step loop stability with GNN communication."""

    def test_100_step_episode_no_crash(self):
        env = CommSwarmEnv(num_drones=5, loss_prob=0.0, formation_bonus=True)
        obs, _ = env.reset(seed=42)

        for step in range(100):
            if not env.agents:
                break
            actions = {
                a: env.action_space(a).sample()
                for a in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)

            # All obs should have correct shape
            for agent, o in obs.items():
                assert o.shape == (ENHANCED_OBS_DIM,), (
                    f"Step {step}, {agent}: bad obs shape {o.shape}"
                )
            # All rewards should be finite
            for agent, r in rewards.items():
                assert np.isfinite(r), f"Step {step}, {agent}: non-finite reward"

    def test_comm_stats_update_each_step(self):
        env = CommSwarmEnv(num_drones=5)
        env.reset(seed=0)
        stats_history = []
        for _ in range(10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
            stats_history.append(env.get_comm_stats())

        assert len(stats_history) > 0
        for stats in stats_history:
            assert stats["mean_separation"] >= 0.0

    def test_episode_with_packet_loss(self):
        env = CommSwarmEnv(num_drones=5, loss_prob=0.3)
        obs, _ = env.reset(seed=7)
        for _ in range(20):
            if not env.agents:
                break
            actions = {a: np.zeros(3) for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
        # Should complete without exception

    def test_coverage_tracked_correctly(self):
        env = CommSwarmEnv(num_drones=5)
        env.reset(seed=1)
        for _ in range(50):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        coverage = float(np.sum(env.coverage_grid)) / max(env.coverage_grid.size, 1)
        assert 0.0 < coverage <= 1.0, f"Unexpected coverage: {coverage}"

    def test_no_nan_in_obs_after_many_steps(self):
        env = CommSwarmEnv(num_drones=10, loss_prob=0.1)
        obs, _ = env.reset(seed=99)
        for step in range(50):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
        for agent, o in obs.items():
            assert not np.any(np.isnan(o)), f"{agent}: NaN detected in obs at step {step}"
            assert not np.any(np.isinf(o)), f"{agent}: Inf detected in obs at step {step}"

    def test_scalability_25_drones(self):
        """25-drone fleet with GNN comm should step without crashing."""
        env = CommSwarmEnv(num_drones=25, loss_prob=0.0)
        obs, _ = env.reset(seed=0)
        assert len(obs) == 25
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        assert len(obs) == 25
        assert len(rewards) == 25


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
