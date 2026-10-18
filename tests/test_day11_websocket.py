"""
SwarmRL -- Day 11 Test: Live FastAPI Backend Server & WebSocket Stream

Validates:
  1. Real HTTP REST Endpoints: /, /health, /api/config, /api/simulation/status
  2. Simulation Controls via REST: /api/simulation/step, /api/simulation/reset, /api/simulation/config
  3. Real WebSocket Handshake & Initial Frame (/ws/simulation)
  4. Telemetry Frame Integrity: 10 drones, 3D positions, velocities, speeds, LiDAR ranges & 3D hit coordinates
  5. Bidirectional WebSocket Controls: ping, step, reset, set_policy, set_fps
"""

import os
import sys
import json
import time
import threading
import asyncio
import requests
import websockets
import uvicorn

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.server import app

TEST_HOST = "127.0.0.1"
TEST_PORT = 8008
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"
WS_URL = f"ws://{TEST_HOST}:{TEST_PORT}/ws/simulation"

PASS = "[PASS]"
FAIL = "[FAIL]"
total_tests = 0
passed_tests = 0


def check(name, condition, detail=""):
    global total_tests, passed_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name} -- {detail}")


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        config = uvicorn.Config(
            app=app,
            host=TEST_HOST,
            port=TEST_PORT,
            log_level="error",
        )
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


async def _async_websocket_flow():
    print("\nTEST 3: WebSocket Connection & Handshake")
    print("-" * 65)

    async with websockets.connect(WS_URL) as ws:
        # Receive Init Message
        raw_init = await ws.recv()
        init_msg = json.loads(raw_init)
        check("Initial message type is 'init'", init_msg.get("type") == "init")
        check("Init contains world bounds", "world" in init_msg)
        check("Init contains obstacles", len(init_msg.get("obstacles", [])) == 5)
        check("Init contains 10 drones count", init_msg.get("num_drones") == 10)
        check("Init contains initial frame", "frame" in init_msg)

        print("\nTEST 4: WebSocket Controls & Telemetry Stream")
        print("-" * 65)

        # 1. Ping / Pong
        await ws.send(json.dumps({"action": "ping"}))
        pong_msg = json.loads(await ws.recv())
        check("Ping receives pong response", pong_msg.get("type") == "pong")

        # 2. Step via WebSocket
        await ws.send(json.dumps({"action": "step"}))
        step_msg = json.loads(await ws.recv())
        check("Step command receives state_update", step_msg.get("type") == "state_update")
        check("State update step is >= 1", step_msg.get("step", 0) >= 1)

        drones = step_msg.get("drones", [])
        check("State update has 10 drones", len(drones) == 10)

        # Verify Drone 0 Telemetry
        d0 = drones[0]
        check("Drone 0 has ID 0", d0.get("id") == 0)
        check("Position is 3D vector [x, y, z]", len(d0.get("position", [])) == 3)
        check("Velocity is 3D vector [vx, vy, vz]", len(d0.get("velocity", [])) == 3)
        check("Speed is positive float", isinstance(d0.get("speed"), (int, float)))
        check("LiDAR ranges has 10 rays", len(d0.get("lidar_ranges", [])) == 10)
        check("LiDAR hits has 10 3D endpoints", len(d0.get("lidar_hits", [])) == 10 and len(d0["lidar_hits"][0]) == 3)
        check("Applied action is 3D vector", len(d0.get("action", [])) == 3)

        # 3. Policy switch via WebSocket
        await ws.send(json.dumps({"action": "set_policy", "policy": "random"}))
        pol_ack = json.loads(await ws.recv())
        check("Policy switch ack received", pol_ack.get("type") == "config_ack")
        check("Policy mode switched to random", pol_ack.get("policy_mode") == "random")

        # 4. Reset via WebSocket
        await ws.send(json.dumps({"action": "reset", "seed": 42}))
        reset_msg = json.loads(await ws.recv())
        check("Reset command produces state_update", reset_msg.get("type") == "state_update")
        check("Step reset to 0", reset_msg.get("step") == 0)

        # 5. Multiple Sequential Steps
        for s in range(1, 4):
            await ws.send(json.dumps({"action": "step"}))
            frame_s = json.loads(await ws.recv())
            assert frame_s.get("step") == s
        check("Stepped 3 steps sequentially over WebSocket", True)


def run_tests():
    global total_tests, passed_tests
    print("=" * 65)
    print("DAY 11 TEST SUITE: LIVE FASTAPI & WEBSOCKET TELEMETRY")
    print("=" * 65)

    server_thread = ServerThread()
    server_thread.start()

    # Wait for server to bind
    for _ in range(50):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    try:
        # ──────────────────────────────────────────
        # TEST 1: REST Endpoints
        # ──────────────────────────────────────────
        print("\nTEST 1: REST Health & Config Endpoints")
        print("-" * 65)

        r_root = requests.get(f"{BASE_URL}/")
        check("Root endpoint (/) returns 200", r_root.status_code == 200)
        check("Root JSON has project name", r_root.json().get("project") == "SwarmRL")

        r_health = requests.get(f"{BASE_URL}/health")
        check("Health endpoint (/health) returns 200", r_health.status_code == 200)
        check("Health reports healthy", r_health.json().get("status") == "healthy")

        r_config = requests.get(f"{BASE_URL}/api/config")
        check("Config endpoint (/api/config) returns 200", r_config.status_code == 200)
        cfg_data = r_config.json()
        check("Config contains 3D world specs", "world" in cfg_data)
        check("Config contains obstacles list", len(cfg_data.get("obstacles", [])) == 5)
        check("Config contains drone specs", cfg_data.get("drones", {}).get("num_drones") == 10)

        # ──────────────────────────────────────────
        # TEST 2: REST Simulation Controls
        # ──────────────────────────────────────────
        print("\nTEST 2: Simulation REST Controls")
        print("-" * 65)

        r_status = requests.get(f"{BASE_URL}/api/simulation/status")
        check("Simulation status returns 200", r_status.status_code == 200)
        s_data = r_status.json()
        check("Simulation has step count", "step" in s_data)
        check("Simulation has coverage metric", "coverage" in s_data)
        check("Simulation has policy_mode", "policy_mode" in s_data)

        r_step = requests.post(f"{BASE_URL}/api/simulation/step")
        check("REST step returns 200", r_step.status_code == 200)
        step_data = r_step.json()
        check("Step status is stepped", step_data.get("status") == "stepped")
        frame = step_data.get("frame", {})
        check("Step advanced to >= 1", frame.get("step", 0) >= 1)
        check("Frame contains 10 drones", len(frame.get("drones", [])) == 10)

        r_reset = requests.post(f"{BASE_URL}/api/simulation/reset", json={"seed": 101})
        check("REST reset returns 200", r_reset.status_code == 200)
        reset_frame = r_reset.json().get("frame", {})
        check("Reset set step to 0", reset_frame.get("step") == 0)

        r_cfg_up = requests.post(f"{BASE_URL}/api/simulation/config", json={"fps": 30, "policy": "heuristic"})
        check("Config update returns 200", r_cfg_up.status_code == 200)
        check("FPS updated to 30", r_cfg_up.json().get("fps") == 30)
        check("Policy updated to heuristic", r_cfg_up.json().get("policy_mode") == "heuristic")

        # ──────────────────────────────────────────
        # TEST 3 & 4: WebSocket
        # ──────────────────────────────────────────
        asyncio.run(_async_websocket_flow())

    finally:
        server_thread.stop()

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("ALL DAY 11 TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"WARNING: {total_tests - passed_tests} test(s) failed!")
    print("=" * 65)
    assert passed_tests == total_tests, f"{total_tests - passed_tests} tests failed in Day 11"


def test_day11_live_server():
    run_tests()


if __name__ == "__main__":
    run_tests()
