"""
SwarmRL -- Day 19: Architecture Ablation Tests

Comprehensive test suite covering:
  Suite 1: Ablation Configuration Matrix (Variants, parameters, compatibility)
  Suite 2: Single Variant Execution (Full Stack, No Comm, MLP Critic, No PopArt)
  Suite 3: Visualization & Technical Report Generation (PNG plot & Markdown report)
  Suite 4: Long-Horizon Curriculum Training & Checkpoint Serialization
  Suite 5: PolicyManager Production Inference Integration
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

from env.swarm_env import SwarmEnv
from env.comm_swarm_env import CommSwarmEnv
from backend.simulation import PolicyManager
from training.ablation_study import (
    ABLATION_VARIANTS,
    run_single_variant_training,
    run_full_ablation_study,
    plot_ablation_comparisons,
    generate_ablation_markdown_report,
)
from training.train_long_horizon import train_long_horizon


# ══════════════════════════════════════════════════════════════
# SUITE 1 — Ablation Configuration Matrix
# ══════════════════════════════════════════════════════════════

class TestAblationMatrixConfig:
    """Validates structural completeness and parameters of ablation configurations."""

    def test_variants_completeness(self):
        expected_keys = {"full_stack_mappo", "no_comm", "mlp_critic", "no_popart"}
        assert set(ABLATION_VARIANTS.keys()) == expected_keys

    @pytest.mark.parametrize("v_key", list(ABLATION_VARIANTS.keys()))
    def test_variant_attributes(self, v_key):
        v = ABLATION_VARIANTS[v_key]
        assert "name" in v
        assert "label" in v
        assert "color" in v
        assert "use_comm" in v
        assert "critic_type" in v
        assert "use_popart" in v
        assert "description" in v

    def test_ablation_specific_differences(self):
        # Baseline: GNN comm, attention critic, popart
        b = ABLATION_VARIANTS["full_stack_mappo"]
        assert b["use_comm"] is True
        assert b["critic_type"] == "attention"
        assert b["use_popart"] is True

        # Ablation A: No comm
        a_comm = ABLATION_VARIANTS["no_comm"]
        assert a_comm["use_comm"] is False
        assert a_comm["critic_type"] == "attention"
        assert a_comm["use_popart"] is True

        # Ablation B: MLP critic
        a_mlp = ABLATION_VARIANTS["mlp_critic"]
        assert a_mlp["use_comm"] is True
        assert a_mlp["critic_type"] == "mlp"
        assert a_mlp["use_popart"] is True

        # Ablation C: No PopArt
        a_pop = ABLATION_VARIANTS["no_popart"]
        assert a_pop["use_comm"] is True
        assert a_pop["critic_type"] == "attention"
        assert a_pop["use_popart"] is False


# ══════════════════════════════════════════════════════════════
# SUITE 2 — Single Variant Execution
# ══════════════════════════════════════════════════════════════

class TestSingleVariantExecution:
    """Validates training execution and telemetry logging of individual ablation variants."""

    def test_run_full_stack_mappo_micro(self):
        res = run_single_variant_training(
            variant_key="full_stack_mappo",
            num_drones=5,
            iterations=2,
            steps_per_iter=10,
        )
        assert res["variant_key"] == "full_stack_mappo"
        assert res["iterations"] == 2
        assert len(res["history"]["coverage"]) == 2
        assert res["final_coverage"] >= 0.0
        assert not np.isnan(res["mean_reward"])
        assert res["total_training_seconds"] > 0.0

    def test_run_mlp_critic_micro(self):
        res = run_single_variant_training(
            variant_key="mlp_critic",
            num_drones=5,
            iterations=2,
            steps_per_iter=10,
        )
        assert res["variant_key"] == "mlp_critic"
        assert len(res["history"]["value_loss"]) == 2
        assert not np.isnan(res["history"]["value_loss"][-1])


# ══════════════════════════════════════════════════════════════
# SUITE 3 — Visualization & Technical Report Generation
# ══════════════════════════════════════════════════════════════

class TestAblationArtifacts:
    """Validates automated plotting and markdown report generation."""

    @pytest.fixture
    def mock_ablation_results(self):
        results = {}
        for k in ABLATION_VARIANTS:
            v_info = ABLATION_VARIANTS[k]
            results[k] = {
                "variant_key": k,
                "name": v_info["name"],
                "label": v_info["label"],
                "color": v_info["color"],
                "final_coverage": 0.25 if k == "full_stack_mappo" else 0.18,
                "mean_reward": 5.2,
                "total_collisions": 3,
                "collisions_per_drone": 0.3,
                "mean_explained_variance": 0.45,
                "total_training_seconds": 2.1,
                "history": {
                    "iterations": [1, 2, 3],
                    "coverage": [0.05, 0.15, 0.25],
                    "explained_variance": [0.1, 0.3, 0.45],
                    "collisions": [1, 1, 1],
                    "rewards": [1.0, 3.0, 5.0],
                },
            }
        return results

    def test_plot_ablation_comparisons(self, mock_ablation_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = os.path.join(tmpdir, "test_ablation.png")
            plot_ablation_comparisons(mock_ablation_results, output_path=out_png)

            assert os.path.exists(out_png)
            assert os.path.getsize(out_png) > 20000  # Multi-panel PNG should be > 20 KB

    def test_generate_ablation_markdown_report(self, mock_ablation_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_md = os.path.join(tmpdir, "test_report.md")
            generate_ablation_markdown_report(mock_ablation_results, output_path=out_md)

            assert os.path.exists(out_md)
            with open(out_md, "r", encoding="utf-8") as f:
                content = f.read()

            assert "# SwarmRL — Multi-Agent Architecture Ablation Study" in content
            assert "Full Stack MAPPO" in content
            assert "No Communication" in content
            assert "MLP Critic" in content
            assert "No PopArt" in content


# ══════════════════════════════════════════════════════════════
# SUITE 4 — Long-Horizon Curriculum Training & Checkpointing
# ══════════════════════════════════════════════════════════════

class TestLongHorizonTraining:
    """Validates curriculum-guided training loop and best model serialization."""

    def test_train_long_horizon_micro(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_best.pt")
            peak_cov = train_long_horizon(
                num_drones=5,
                total_iterations=2,
                steps_per_iter=12,
                checkpoint_output=ckpt_path,
            )

            assert peak_cov >= 0.0
            assert os.path.exists(ckpt_path)

            # Inspect checkpoint contents
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert "actor_state_dict" in ckpt
            assert "critic_state_dict" in ckpt
            assert "iteration" in ckpt


# ══════════════════════════════════════════════════════════════
# SUITE 5 — PolicyManager Production Inference Integration
# ══════════════════════════════════════════════════════════════

class TestProductionInferenceIntegration:
    """Validates that PolicyManager loads long-horizon checkpoints and executes inference."""

    def test_policy_manager_loads_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "day19_mappo_best.pt")
            train_long_horizon(
                num_drones=4,
                total_iterations=1,
                steps_per_iter=10,
                checkpoint_output=ckpt_path,
            )

            pm = PolicyManager(checkpoint_path=ckpt_path)
            assert pm.native_mappo_actor is not None
            assert pm.mode == "trained"

            # Execute action inference
            env = CommSwarmEnv(num_drones=4)
            obs, _ = env.reset()
            actions = pm.compute_actions(env, obs)

            assert len(actions) == 4
            for a_name, act in actions.items():
                assert act.shape == (3,)
                assert np.all(act >= -1.0) and np.all(act <= 1.0)
