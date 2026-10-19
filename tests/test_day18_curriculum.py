"""
SwarmRL -- Day 18: Curriculum Learning & Optimization Tests

Comprehensive test suite covering:
  Suite 1: Curriculum Stages & Manager (Stage definition, transitions, environment adaptation)
  Suite 2: Hyperparameter Tuning Harness (Parameter sampling, trial execution, Pareto frontier)
  Suite 3: Fleet Scalability Benchmarks (Variable fleet execution, throughput calculation)
  Suite 4: Native MAPPO PolicyManager Integration (Checkpoint loading, action generation, dimension padding)
  Suite 5: End-to-End Curriculum-Guided MAPPO Training Loop
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch

# ── path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.world_config import WORLD_CONFIG
from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from backend.simulation import PolicyManager
from training.curriculum import CurriculumStage, CurriculumManager
from training.tune_mappo import (
    DEFAULT_PARAM_GRID,
    sample_hyperparameters,
    run_trial,
    compute_pareto_frontier,
)
from training.scale_benchmark import benchmark_fleet_scale
from training.mappo_trainer import MAPPOConfig, MAPPOTrainer
from env.gnn_comm import ENHANCED_OBS_DIM


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def curriculum_mgr():
    return CurriculumManager()


@pytest.fixture
def test_env():
    return CommSwarmEnv(num_drones=5, loss_prob=0.0, formation_bonus=True)


# ══════════════════════════════════════════════════════════════
# SUITE 1 — Curriculum Stages & Manager
# ══════════════════════════════════════════════════════════════

class TestCurriculumManager:
    """Validates curriculum progression, environment parameter adaptation, and promotion criteria."""

    def test_default_stages_structure(self, curriculum_mgr):
        assert len(curriculum_mgr.stages) == 3
        assert curriculum_mgr.current_stage_idx == 0
        s1 = curriculum_mgr.current_stage
        assert s1.stage_id == 1
        assert s1.max_steps == 100
        assert s1.collision_penalty == -4.0
        assert s1.explore_reward == 0.8
        assert not s1.formation_bonus

    def test_apply_to_env(self, curriculum_mgr, test_env):
        curriculum_mgr.apply_to_env(test_env)
        inner = test_env._inner

        assert inner.max_steps == 100
        assert inner.target_coverage == 0.15
        assert inner.reward_explore == 0.8
        assert inner.reward_collision == -4.0
        assert not test_env.formation_bonus

    def test_promotion_sequence(self, curriculum_mgr):
        # Stage 1 promotion requires: coverage >= 0.12, collisions <= 2.0 (3 times)
        pass_metrics_s1 = {"final_coverage": 0.14, "collisions": 1.0}
        fail_metrics_s1 = {"final_coverage": 0.05, "collisions": 5.0}

        # 1. Failure resets counter
        curriculum_mgr.evaluate_progress(pass_metrics_s1)
        assert curriculum_mgr.consecutive_successes == 1
        curriculum_mgr.evaluate_progress(fail_metrics_s1)
        assert curriculum_mgr.consecutive_successes == 0

        # 2. 3 consecutive passes promote to Stage 2
        curriculum_mgr.evaluate_progress(pass_metrics_s1)
        curriculum_mgr.evaluate_progress(pass_metrics_s1)
        promoted, reason = curriculum_mgr.evaluate_progress(pass_metrics_s1)

        assert promoted
        assert curriculum_mgr.current_stage_idx == 1
        assert curriculum_mgr.current_stage.stage_id == 2
        assert "Zone A" in reason

        # 3. Stage 2 promotion requires: coverage >= 0.28, zone_a >= 0.35, collisions <= 4.0
        pass_metrics_s2 = {"final_coverage": 0.32, "zone_a_coverage": 0.40, "collisions": 2.0}
        curriculum_mgr.evaluate_progress(pass_metrics_s2)
        curriculum_mgr.evaluate_progress(pass_metrics_s2)
        promoted_s2, reason_s2 = curriculum_mgr.evaluate_progress(pass_metrics_s2)

        assert promoted_s2
        assert curriculum_mgr.current_stage_idx == 2
        assert curriculum_mgr.current_stage.stage_id == 3
        assert curriculum_mgr.is_final_stage

    def test_status_dict(self, curriculum_mgr):
        status = curriculum_mgr.get_status_dict()
        assert status["stage_id"] == 1
        assert "stage_name" in status
        assert status["max_steps"] == 100
        assert not status["is_final_stage"]


# ══════════════════════════════════════════════════════════════
# SUITE 2 — Hyperparameter Tuning Harness
# ══════════════════════════════════════════════════════════════

class TestHyperparameterTuning:
    """Validates parameter sampling, trial execution, and Pareto frontier ranking."""

    def test_sample_hyperparameters(self):
        samples = sample_hyperparameters(DEFAULT_PARAM_GRID, num_samples=3, seed=123)
        assert len(samples) == 3
        for s in samples:
            assert "lr_actor" in s
            assert "lr_critic" in s
            assert "lr_ratio" in s
            assert s["lr_critic"] == s["lr_actor"] * s["lr_ratio"]
            assert s["entropy_coef"] in DEFAULT_PARAM_GRID["entropy_coef"]

    def test_run_trial(self):
        params = {
            "trial_id": 1,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "lr_ratio": 3.33,
            "entropy_coef": 0.01,
            "clip_param": 0.2,
            "gae_lambda": 0.95,
        }
        res = run_trial(
            params=params,
            num_drones=5,
            iterations=2,
            steps_per_iter=15,
            use_comm=True,
        )
        assert res["trial_id"] == 1
        assert "final_coverage" in res
        assert "mean_reward" in res
        assert "total_collisions" in res
        assert res["duration_seconds"] > 0.0

    def test_compute_pareto_frontier(self):
        results = [
            {"trial_id": 1, "final_coverage": 0.30, "total_collisions": 2, "mean_reward": 5.0},
            {"trial_id": 2, "final_coverage": 0.20, "total_collisions": 5, "mean_reward": 2.0},  # Dominated by 1
            {"trial_id": 3, "final_coverage": 0.35, "total_collisions": 6, "mean_reward": 7.0},  # Higher cov & rew, trade-off cols
        ]
        pareto = compute_pareto_frontier(results)
        pareto_ids = [p["trial_id"] for p in pareto]

        assert 1 in pareto_ids
        assert 3 in pareto_ids
        assert 2 not in pareto_ids  # Trial 2 is strictly dominated


# ══════════════════════════════════════════════════════════════
# SUITE 3 — Fleet Scalability Benchmarking
# ══════════════════════════════════════════════════════════════

class TestFleetScalabilityBenchmark:
    """Validates multi-fleet benchmarking execution and throughput reporting."""

    def test_benchmark_fleet_scale_micro(self):
        results = benchmark_fleet_scale(
            fleet_sizes=[5, 10],
            iterations=1,
            steps_per_iter=10,
        )
        assert "fleet_5" in results
        assert "fleet_10" in results

        f5 = results["fleet_5"]
        f10 = results["fleet_10"]

        assert f5["num_drones"] == 5
        assert f10["num_drones"] == 10
        assert f5["effective_fps"] > 0.0
        assert f10["effective_fps"] > 0.0
        assert f10["total_transitions_per_iter"] == 10 * 10


# ══════════════════════════════════════════════════════════════
# SUITE 4 — Native MAPPO PolicyManager Integration
# ══════════════════════════════════════════════════════════════

class TestPolicyManagerIntegration:
    """Validates native PyTorch MAPPO checkpoint loading and inference inside PolicyManager."""

    def test_policy_manager_loads_native_mappo(self):
        pm = PolicyManager()
        # Should have found and loaded Day 17 checkpoint
        assert pm.native_mappo_actor is not None
        assert pm.mode == "trained"
        assert pm.native_mappo_actor.obs_dim == 42

    def test_policy_manager_action_computation_with_padding(self):
        pm = PolicyManager()
        env = SwarmEnv(num_drones=4)
        obs, _ = env.reset()

        # obs from SwarmEnv has 26 dimensions, PolicyManager should auto-pad to 42
        actions = pm.compute_actions(env, obs)
        assert len(actions) == 4
        for did, act in actions.items():
            assert act.shape == (3,)
            assert np.all(act >= -1.0) and np.all(act <= 1.0)

    def test_policy_manager_mode_switching(self):
        pm = PolicyManager()
        env = SwarmEnv(num_drones=3)
        obs, _ = env.reset()

        # Mode: heuristic
        pm.set_mode("heuristic")
        act_heur = pm.compute_actions(env, obs)
        assert len(act_heur) == 3

        # Mode: random
        pm.set_mode("random")
        act_rand = pm.compute_actions(env, obs)
        assert len(act_rand) == 3

        # Mode: trained
        pm.set_mode("trained")
        assert pm.mode == "trained"


# ══════════════════════════════════════════════════════════════
# SUITE 5 — End-to-End Curriculum-Guided MAPPO Training
# ══════════════════════════════════════════════════════════════

class TestCurriculumTrainingLoop:
    """Validates unified curriculum progression with MAPPO CTDE optimization."""

    def test_curriculum_training_iteration(self, test_env, curriculum_mgr):
        curriculum_mgr.apply_to_env(test_env)

        config = MAPPOConfig(
            lr_actor=3e-4,
            lr_critic=1e-3,
            ppo_epochs=1,
            actor_mini_batch_size=32,
            critic_mini_batch_size=16,
            use_popart=True,
        )
        trainer = MAPPOTrainer(
            config=config,
            obs_dim=ENHANCED_OBS_DIM,
            num_agents=5,
            rollout_steps=10,
        )

        metrics = trainer.train_iteration(test_env, num_steps=10)
        assert "mean_reward" in metrics
        assert "final_coverage" in metrics

        promoted, reason = curriculum_mgr.evaluate_progress(metrics)
        assert isinstance(promoted, bool)
        assert len(curriculum_mgr.stage_history) == 1
