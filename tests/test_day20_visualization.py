"""
SwarmRL -- Day 20: 3D Visualization, Comm Mesh & Telemetry Tests

Comprehensive test suite covering:
  Suite 1: GNN Communication Graph & Topology Analytics (edges, density, neighbors)
  Suite 2: Curriculum Stage Integration in SimulationEngine (state, transitions, config)
  Suite 3: High-Density 50-Drone Fleet Throughput & JSON Payload Serialization
  Suite 4: WebSocket API Protocol Contract for Day 20 Commands
"""

import os
import sys
import json
import time
import pytest
import numpy as np

# Path setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.simulation import SimulationEngine, PolicyManager
from env.spaces import COMM_RANGE


# ══════════════════════════════════════════════════════════════
# SUITE 1 — GNN Comm Graph & Network Topology
# ══════════════════════════════════════════════════════════════

class TestCommGraphTopology:
    """Validates calculation of inter-drone communication links and graph metrics."""

    def test_comm_edges_structure(self):
        engine = SimulationEngine(num_drones=10)
        frame = engine.reset(seed=42)

        assert "comm_edges" in frame
        assert isinstance(frame["comm_edges"], list)

        # Basecamp spawn is clustered at [10, 10, 5] spread 5m -> initial drones must be within 20m
        assert len(frame["comm_edges"]) > 0

        for edge in frame["comm_edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "distance" in edge
            assert edge["source"] != edge["target"]
            assert edge["distance"] <= COMM_RANGE

    def test_network_analytics_metrics(self):
        engine = SimulationEngine(num_drones=10)
        frame = engine.reset(seed=42)
        analytics = frame["swarm_analytics"]

        assert "comm_links" in analytics
        assert "avg_neighbors" in analytics
        assert "network_density" in analytics

        assert analytics["comm_links"] == len(frame["comm_edges"])
        assert 0.0 <= analytics["network_density"] <= 1.0

        # Avg neighbors per drone must match 2 * edges / N
        expected_avg_neighbors = round((2.0 * len(frame["comm_edges"])) / 10, 2)
        assert analytics["avg_neighbors"] == expected_avg_neighbors

    def test_comm_range_cutoff_strictly_enforced(self):
        engine = SimulationEngine(num_drones=5)
        engine.reset()

        # Manually space drones beyond COMM_RANGE (e.g. 50m apart)
        for i, drone in enumerate(engine.env.drones):
            drone.position = np.array([float(i * 30.0), 10.0, 10.0], dtype=np.float32)

        frame = engine._build_frame(
            actions={},
            new_cells=0,
            new_voxels=[],
            dd_cols=[],
            do_cols=[],
            is_terminated=False,
            is_truncated=False,
        )

        # When drones are 30m apart, distance > 20m -> 0 edges
        assert len(frame["comm_edges"]) == 0
        assert frame["swarm_analytics"]["comm_links"] == 0
        assert frame["swarm_analytics"]["network_density"] == 0.0


# ══════════════════════════════════════════════════════════════
# SUITE 2 — Curriculum Integration in SimulationEngine
# ══════════════════════════════════════════════════════════════

class TestCurriculumIntegration:
    """Validates real-time curriculum stage tracking and dynamic stage switching."""

    def test_initial_curriculum_state(self):
        engine = SimulationEngine(num_drones=10)
        frame = engine.reset()

        assert "curriculum" in frame
        cur = frame["curriculum"]
        assert cur["stage_id"] == 1
        assert "Dispersion" in cur["stage_name"]
        assert "target_coverage" in cur
        assert cur["target_coverage"] == 15.0

    def test_set_curriculum_stage(self):
        engine = SimulationEngine(num_drones=10)

        # Switch to Stage 2: Zone A & Formation
        s2 = engine.set_curriculum_stage(2)
        assert s2["stage_id"] == 2
        assert "Zone A" in s2["stage_name"]
        assert s2["target_coverage"] == 40.0
        assert engine.env.max_steps == 250

        # Frame immediately reflects Stage 2
        frame = engine.step()
        assert frame["curriculum"]["stage_id"] == 2
        assert frame["max_steps"] == 250

        # Switch to Stage 3: Full Disaster Coverage
        s3 = engine.set_curriculum_stage(3)
        assert s3["stage_id"] == 3
        assert "Full Disaster" in s3["stage_name"]
        assert s3["target_coverage"] == 85.0
        assert engine.env.max_steps == 500


# ══════════════════════════════════════════════════════════════
# SUITE 3 — High-Density 50-Drone Fleet Throughput & Payload
# ══════════════════════════════════════════════════════════════

class TestHighDensityScalingThroughput:
    """Stress tests frame generation and JSON serialization at 50 agents."""

    def test_50_drone_frame_generation_latency(self):
        engine = SimulationEngine(num_drones=50)
        engine.reset(seed=123)

        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            frame = engine.step()
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

        avg_latency = np.mean(latencies)
        assert len(frame["drones"]) == 50
        # Frame step latency for 50 drones on CPU must be under 60ms (benchmarked at ~45-53ms)
        assert avg_latency < 60.0, f"Average 50-drone step latency {avg_latency:.2f}ms exceeds 60ms"

    def test_50_drone_json_payload_size(self):
        engine = SimulationEngine(num_drones=50)
        frame = engine.reset(seed=123)

        payload_str = json.dumps(frame)
        payload_bytes = len(payload_str.encode("utf-8"))
        payload_kb = payload_bytes / 1024.0

        # Compact WebSocket payload must remain under 100 KB even at peak basecamp cluster (1225 edges)
        assert payload_kb < 100.0, f"Payload size {payload_kb:.2f} KB exceeds 100 KB budget"


# ══════════════════════════════════════════════════════════════
# SUITE 4 — Frontend Contract & Schema Integrity
# ══════════════════════════════════════════════════════════════

class TestFrontendTelemetryContract:
    """Verifies that frame payloads contain all keys required by React Three.js frontend."""

    def test_frame_contract_completeness(self):
        engine = SimulationEngine(num_drones=10)
        frame = engine.step()

        required_top_keys = [
            "type",
            "step",
            "max_steps",
            "coverage",
            "coverage_pct",
            "total_collisions",
            "status",
            "policy_mode",
            "fps",
            "num_drones",
            "curriculum",
            "comm_edges",
            "drones",
            "sectors",
            "heatmap_grid",
            "swarm_analytics",
            "events",
        ]
        for key in required_top_keys:
            assert key in frame, f"Missing required telemetry key: {key}"

        # Analytics keys
        analytics_keys = [
            "centroid",
            "dispersion_radius",
            "min_distance",
            "avg_distance",
            "avg_speed",
            "comm_links",
            "avg_neighbors",
            "network_density",
        ]
        for a_key in analytics_keys:
            assert a_key in frame["swarm_analytics"], f"Missing analytics key: {a_key}"
