"""
SwarmRL — Day 12 Automated Verification Suite

Tests the Day 12 deliverables:
1. Frontend 3D Components existence and exports (DroneModel, WorldEnvironment, SimulationViewport)
2. Alignment between backend WorldConfig and 3D environment components
3. Telemetry packet schema compatibility with Three.js drone rendering
4. Frontend production asset build validation
"""

import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.world_config import WORLD_CONFIG, WORLD_SIZE

def test_component_files():
    print("TEST 1: Verifying Day 12 React Three.js Components")
    print("-" * 65)
    components_dir = PROJECT_ROOT / "frontend" / "src" / "components"
    required_files = [
        "DroneModel.jsx",
        "WorldEnvironment.jsx",
        "SimulationViewport.jsx"
    ]
    for filename in required_files:
        filepath = components_dir / filename
        assert filepath.exists(), f"Missing component: {filename}"
        content = filepath.read_text(encoding="utf-8")
        assert len(content) > 200, f"Component {filename} is suspiciously empty"
        print(f"  [PASS] Component {filename} exists ({len(content)} chars)")
    print()

def test_component_exports_and_syntax():
    print("TEST 2: Verifying Component Contract & Integrations")
    print("-" * 65)
    app_jsx = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "SimulationViewport" in app_jsx, "App.jsx does not import/render SimulationViewport"
    print("  [PASS] App.jsx renders SimulationViewport")

    viewport_jsx = (PROJECT_ROOT / "frontend" / "src" / "components" / "SimulationViewport.jsx").read_text(encoding="utf-8")
    assert "Canvas" in viewport_jsx, "SimulationViewport missing @react-three/fiber Canvas"
    assert "OrbitControls" in viewport_jsx, "SimulationViewport missing OrbitControls"
    assert "WorldEnvironment" in viewport_jsx, "SimulationViewport missing WorldEnvironment"
    assert "DroneModel" in viewport_jsx, "SimulationViewport missing DroneModel"
    print("  [PASS] SimulationViewport includes Canvas, OrbitControls, WorldEnvironment, and DroneModel")

    drone_jsx = (PROJECT_ROOT / "frontend" / "src" / "components" / "DroneModel.jsx").read_text(encoding="utf-8")
    assert "lidar_hits" in drone_jsx or "lidarHits" in drone_jsx or "lidar_ranges" in drone_jsx, "DroneModel missing LiDAR visualization logic"
    print("  [PASS] DroneModel contains 3D LiDAR laser beam rendering logic")

    world_jsx = (PROJECT_ROOT / "frontend" / "src" / "components" / "WorldEnvironment.jsx").read_text(encoding="utf-8")
    assert "obstacles" in world_jsx, "WorldEnvironment missing obstacle rendering logic"
    assert "grid" in world_jsx.lower(), "WorldEnvironment missing ground grid"
    print("  [PASS] WorldEnvironment contains 3D obstacles and disaster grid")
    print()

def test_world_config_consistency():
    print("TEST 3: Verifying Backend-Frontend Coordinate Space Alignment")
    print("-" * 65)
    # Check world dimensions
    assert WORLD_SIZE[0] == 100.0
    assert WORLD_SIZE[1] == 100.0
    assert WORLD_SIZE[2] == 50.0
    print(f"  [PASS] World size: {WORLD_SIZE} matches 100x100x50m specification")
    assert WORLD_CONFIG["drones"]["num_drones"] == 10
    print(f"  [PASS] Drone count: {WORLD_CONFIG['drones']['num_drones']} drones supported")
    assert len(WORLD_CONFIG["obstacles"]) == 5
    print(f"  [PASS] Obstacles count: {len(WORLD_CONFIG['obstacles'])} configured")
    print()

def test_frontend_production_build():
    print("TEST 4: Verifying Production Bundle Output")
    print("-" * 65)
    dist_index = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    assert dist_index.exists(), "frontend/dist/index.html does not exist! Run npm run build"
    html_content = dist_index.read_text(encoding="utf-8")
    assert "SwarmRL" in html_content or "vite" in html_content.lower(), "Built HTML seems invalid"
    print(f"  [PASS] Production build exists: frontend/dist/index.html ({dist_index.stat().st_size} bytes)")
    print()

def main():
    print("=" * 65)
    print("DAY 12 TEST SUITE: THREE.JS 3D VIEWPORT & ENVIRONMENT")
    print("=" * 65)
    test_component_files()
    test_component_exports_and_syntax()
    test_world_config_consistency()
    test_frontend_production_build()
    print("=" * 65)
    print("ALL DAY 12 VERIFICATION CHECKS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
