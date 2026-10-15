"""
SwarmRL — Day 14 Automated Verification Suite

Tests the Day 14 deliverables:
1. Dynamic multi-agent scalability in SwarmEnv (10, 25, 50 drones)
2. SimulationEngine fleet scaling via set_num_drones()
3. Benchmark evaluation runner and generated artifacts:
   - evaluation_results.json
   - scalability_results.json
   - policy_comparison.png
   - scalability_comparison.png
   - BENCHMARK_REPORT.md
4. Frontend production distribution build validation
"""

import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.swarm_env import SwarmEnv
from backend.simulation import SimulationEngine


def test_swarm_env_scalability():
    print("TEST 1: Verifying SwarmEnv Multi-Agent Scalability (10, 25, 50 Drones)")
    print("-" * 65)
    for n in [10, 25, 50]:
        env = SwarmEnv(num_drones=n)
        assert env.num_drones == n, f"Expected {n} drones, got {env.num_drones}"
        assert len(env.possible_agents) == n, f"Expected {n} possible agents"
        obs, infos = env.reset(seed=42)
        assert len(env.drones) == n, f"Expected {n} drone instances"
        assert len(obs) == n, f"Expected {n} observations"

        # Step with zero actions
        actions = {f"drone_{i}": np.zeros(3) for i in range(n)}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        assert len(next_obs) == n, f"Step did not return {n} observations"
        print(f"  [PASS] SwarmEnv successfully initialized & stepped with {n} drones")
    print()


def test_simulation_engine_scaling():
    print("TEST 2: Verifying SimulationEngine Dynamic Fleet Scaling")
    print("-" * 65)
    engine = SimulationEngine(fps=30, num_drones=10)
    assert engine.num_drones == 10
    assert len(engine.latest_frame["drones"]) == 10
    print("  [PASS] Initial fleet size: 10 drones")

    # Scale up to 25
    engine.set_num_drones(25)
    assert engine.num_drones == 25
    assert len(engine.latest_frame["drones"]) == 25
    assert engine.latest_frame["num_drones"] == 25
    frame = engine.step()
    assert len(frame["drones"]) == 25
    print("  [PASS] Dynamically scaled to 25 drones and verified step execution")

    # Scale up to 50
    engine.set_num_drones(50)
    assert engine.num_drones == 50
    assert len(engine.latest_frame["drones"]) == 50
    assert engine.latest_frame["num_drones"] == 50
    frame = engine.step()
    assert len(frame["drones"]) == 50
    print("  [PASS] Dynamically scaled to 50 drones and verified step execution")
    print()


def test_benchmark_artifacts():
    print("TEST 3: Verifying Benchmark Artifacts & Report Files")
    print("-" * 65)
    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    required_files = [
        "evaluate.py",
        "evaluation_results.json",
        "scalability_results.json",
        "policy_comparison.png",
        "scalability_comparison.png",
        "BENCHMARK_REPORT.md",
    ]

    for fname in required_files:
        fpath = benchmarks_dir / fname
        assert fpath.exists(), f"Missing required benchmark artifact: {fname}"
        size = fpath.stat().st_size
        assert size > 100, f"Artifact {fname} is unexpectedly small ({size} bytes)"
        print(f"  [PASS] Benchmark file exists: {fname} ({size} bytes)")

    # Validate JSON structure
    with open(benchmarks_dir / "evaluation_results.json") as f:
        eval_data = json.load(f)
        for pol in ["trained", "heuristic", "random"]:
            assert pol in eval_data, f"Evaluation JSON missing policy: {pol}"
            assert "mean_final_coverage" in eval_data[pol]
            assert "mean_collisions" in eval_data[pol]
    print("  [PASS] evaluation_results.json verified with MAPPO, Heuristic, and Random metrics")

    with open(benchmarks_dir / "scalability_results.json") as f:
        scale_data = json.load(f)
        for n_str in ["10", "25", "50"]:
            assert n_str in scale_data, f"Scalability JSON missing fleet size: {n_str}"
            assert "throughput_fps" in scale_data[n_str]
            assert "mean_step_latency_ms" in scale_data[n_str]
    print("  [PASS] scalability_results.json verified across 10, 25, and 50 drone fleets")
    print()


def test_production_bundle():
    print("TEST 4: Verifying Production Frontend Build")
    print("-" * 65)
    dist_html = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    assert dist_html.exists(), "frontend/dist/index.html is missing"
    print(f"  [PASS] Production bundle verified: dist/index.html ({dist_html.stat().st_size} bytes)")
    print()


def main():
    print("=" * 65)
    print("DAY 14 TEST SUITE: MULTI-AGENT SCALABILITY & EVALUATION")
    print("=" * 65)
    test_swarm_env_scalability()
    test_simulation_engine_scaling()
    test_benchmark_artifacts()
    test_production_bundle()
    print("=" * 65)
    print("ALL DAY 14 VERIFICATION CHECKS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
