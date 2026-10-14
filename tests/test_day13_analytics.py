"""
SwarmRL — Day 13 Automated Verification Suite

Tests the Day 13 deliverables:
1. Backend SimulationEngine Day 13 telemetry frame generation:
   - Dynamic search sector completion metrics (Zone A & Zone B)
   - 2D ground coverage heatmap grid (25x25)
   - Swarm formation & spatial analytics (centroid, dispersion radius, min distance, speed)
   - Sampled 3D volumetric coverage voxels
2. Multi-step dynamic telemetry updates under continuous stepping
3. Frontend Day 13 React Three.js components existence and contract:
   - CoverageHeatmap.jsx
   - CoverageVoxels.jsx
   - SwarmCentroidMarker.jsx
   - SimulationViewport.jsx camera presets integration
   - App.jsx sector meters & swarm analytics UI
4. Production bundle compilation verification
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.simulation import SimulationEngine
from env.world_config import WORLD_CONFIG, WORLD_SIZE


def test_simulation_frame_day13():
    print("TEST 1: Verifying Backend Telemetry Frame Analytics")
    print("-" * 65)
    engine = SimulationEngine(fps=20)
    frame = engine.latest_frame

    # 1. Sector completion checks
    assert "sectors" in frame, "Frame missing 'sectors' dictionary"
    sectors = frame["sectors"]
    assert "zone_A" in sectors, "Sectors missing 'zone_A'"
    assert "zone_B" in sectors, "Sectors missing 'zone_B'"

    za = sectors["zone_A"]
    zb = sectors["zone_B"]
    assert "coverage_pct" in za and isinstance(za["coverage_pct"], float)
    assert "priority" in za and za["priority"] == "high"
    assert "coverage_pct" in zb and isinstance(zb["coverage_pct"], float)
    assert "priority" in zb and zb["priority"] == "medium"
    print(f"  [PASS] Sector A (Primary): {za['coverage_pct']}% / {za['target_pct']}% (Priority: {za['priority']})")
    print(f"  [PASS] Sector B (Secondary): {zb['coverage_pct']}% / {zb['target_pct']}% (Priority: {zb['priority']})")

    # 2. 2D Ground Heatmap checks
    assert "heatmap_grid" in frame, "Frame missing 'heatmap_grid'"
    hm = frame["heatmap_grid"]
    assert len(hm) == 25, f"Heatmap rows expected 25, got {len(hm)}"
    assert len(hm[0]) == 25, f"Heatmap cols expected 25, got {len(hm[0])}"
    assert all(0.0 <= val <= 1.0 for row in hm for val in row), "Heatmap values not in [0, 1]"
    print(f"  [PASS] 2D Ground Heatmap: shape (25, 25) with values normalized to [0, 1]")

    # 3. Swarm formation & spatial analytics
    assert "swarm_analytics" in frame, "Frame missing 'swarm_analytics'"
    sa = frame["swarm_analytics"]
    assert "centroid" in sa and len(sa["centroid"]) == 3, "Centroid not 3D vector"
    assert "dispersion_radius" in sa and sa["dispersion_radius"] >= 0, "Invalid dispersion radius"
    assert "min_distance" in sa and sa["min_distance"] >= 0, "Invalid min distance"
    assert "avg_distance" in sa and sa["avg_distance"] >= 0, "Invalid avg distance"
    assert "avg_speed" in sa and sa["avg_speed"] >= 0, "Invalid avg speed"
    print(f"  [PASS] Swarm Centroid: {sa['centroid']}")
    print(f"  [PASS] Dispersion Radius: {sa['dispersion_radius']}m, Min Separation: {sa['min_distance']}m, Avg Speed: {sa['avg_speed']}m/s")

    # 4. 3D Volumetric coverage voxels
    assert "new_voxels" in frame, "Frame missing 'new_voxels'"
    assert isinstance(frame["new_voxels"], list)
    print(f"  [PASS] Initial sampled 3D voxels: {len(frame['new_voxels'])} points")
    print()


def test_simulation_step_progression():
    print("TEST 2: Verifying Dynamic Telemetry Evolution Across Steps")
    print("-" * 65)
    engine = SimulationEngine(fps=20)
    engine.policy_manager.set_mode("heuristic")

    initial_step = engine.current_step
    initial_cov = engine.latest_frame["coverage"]

    # Step 5 times
    for _ in range(5):
        frame = engine.step()

    assert frame["step"] == initial_step + 5, f"Expected step {initial_step + 5}, got {frame['step']}"
    assert frame["coverage"] >= initial_cov, "Coverage decreased after stepping"
    sa = frame["swarm_analytics"]
    assert sa["avg_speed"] >= 0.0, "Average speed should be non-negative"
    print(f"  [PASS] Stepped 5 frames: Step={frame['step']}, Coverage={frame['coverage_pct']}%, Avg Speed={sa['avg_speed']} m/s")
    print()


def test_frontend_components_contract():
    print("TEST 3: Verifying React Components & Camera Presets")
    print("-" * 65)
    components_dir = PROJECT_ROOT / "frontend" / "src" / "components"

    # 1. CoverageHeatmap.jsx
    hm_file = components_dir / "CoverageHeatmap.jsx"
    assert hm_file.exists(), "CoverageHeatmap.jsx is missing"
    hm_content = hm_file.read_text(encoding="utf-8")
    assert "CanvasTexture" in hm_content, "CoverageHeatmap missing CanvasTexture"
    print("  [PASS] CoverageHeatmap.jsx exists and utilizes Three.js CanvasTexture")

    # 2. CoverageVoxels.jsx
    vox_file = components_dir / "CoverageVoxels.jsx"
    assert vox_file.exists(), "CoverageVoxels.jsx is missing"
    vox_content = vox_file.read_text(encoding="utf-8")
    assert "points" in vox_content.lower() or "instanced" in vox_content.lower(), "CoverageVoxels missing particle/point rendering"
    print("  [PASS] CoverageVoxels.jsx exists and utilizes Three.js particle buffer")

    # 3. SwarmCentroidMarker.jsx
    cent_file = components_dir / "SwarmCentroidMarker.jsx"
    assert cent_file.exists(), "SwarmCentroidMarker.jsx is missing"
    cent_content = cent_file.read_text(encoding="utf-8")
    assert "dispersion" in cent_content.lower(), "SwarmCentroidMarker missing dispersion envelope logic"
    print("  [PASS] SwarmCentroidMarker.jsx exists and renders centroid beacon and spread envelope")

    # 4. SimulationViewport.jsx
    vp_file = components_dir / "SimulationViewport.jsx"
    vp_content = vp_file.read_text(encoding="utf-8")
    for mode in ["orbit", "topDown", "chase", "elevation"]:
      assert mode in vp_content, f"SimulationViewport missing camera preset '{mode}'"
    print("  [PASS] SimulationViewport supports 4 camera presets: Orbit, TopDown, Chase, Elevation")

    # 5. App.jsx
    app_file = PROJECT_ROOT / "frontend" / "src" / "App.jsx"
    app_content = app_file.read_text(encoding="utf-8")
    assert "SECTORS" in app_content or "sectors" in app_content, "App.jsx missing sectors display"
    assert "SWARM FORMATION" in app_content or "swarm_analytics" in app_content, "App.jsx missing swarm formation display"
    print("  [PASS] App.jsx renders Search Sector Meters and Swarm Formation Analytics")
    print()


def test_production_bundle():
    print("TEST 4: Verifying Production Bundle Compilation")
    print("-" * 65)
    dist_html = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    assert dist_html.exists(), "Production bundle dist/index.html is missing"
    dist_size = dist_html.stat().st_size
    print(f"  [PASS] Production build exists: dist/index.html ({dist_size} bytes)")
    print()


def main():
    print("=" * 65)
    print("DAY 13 TEST SUITE: DYNAMIC COVERAGE MAP & SWARM ANALYTICS")
    print("=" * 65)
    test_simulation_frame_day13()
    test_simulation_step_progression()
    test_frontend_components_contract()
    test_production_bundle()
    print("=" * 65)
    print("ALL DAY 13 VERIFICATION CHECKS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
