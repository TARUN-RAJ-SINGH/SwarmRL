"""
SwarmRL -- Day 11: FastAPI Backend Server & WebSocket Stream

Serves as the high-throughput bridge between the RL simulation engine and the React frontend.
Provides:
  - WebSocket endpoint (/ws/simulation) streaming drone telemetry at 20-30 Hz
  - Full bidirectional control (start, pause, reset, step, set_fps, set_policy)
  - REST endpoints for configuration, status, and health checks
"""

import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.world_config import WORLD_CONFIG
from backend.simulation import SimulationEngine

# Global simulation engine instance
engine: Optional[SimulationEngine] = None
connected_clients: list[WebSocket] = []


async def broadcast_frame(frame: dict):
    """Broadcast simulation frame to all active WebSocket clients."""
    if not connected_clients:
        return

    dead_clients = []
    # Use JSON string serialization for fast transmission
    payload = json.dumps(frame)
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead_clients.append(ws)

    for dead in dead_clients:
        if dead in connected_clients:
            connected_clients.remove(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and teardown simulation engine."""
    global engine
    print("[Server] Initializing SimulationEngine...")
    engine = SimulationEngine(fps=20)
    engine.register_broadcast_callback(broadcast_frame)
    print(f"[Server] SimulationEngine ready. Initial policy: {engine.policy_manager.mode}")
    yield
    print("[Server] Shutting down simulation engine...")
    if engine:
        engine.pause()


app = FastAPI(
    title="SwarmRL Backend",
    description="Real-time multi-agent drone swarm simulation and inference server",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Request Models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None


class ConfigUpdateRequest(BaseModel):
    fps: Optional[int] = None
    policy: Optional[str] = None
    num_drones: Optional[int] = None


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "project": "SwarmRL",
        "version": "0.2.0",
        "status": "online",
        "active_clients": len(connected_clients),
        "policy_mode": engine.policy_manager.mode if engine else "none",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "sim_status": engine.status if engine else "offline",
        "active_clients": len(connected_clients),
    }


@app.get("/api/config")
async def get_config():
    """Return complete simulation world and obstacle configuration."""
    return WORLD_CONFIG


@app.get("/api/simulation/status")
async def get_simulation_status():
    """Return current simulation metrics and status."""
    if not engine:
        return {"error": "Engine not initialized"}

    return {
        "status": engine.status,
        "step": engine.current_step,
        "max_steps": engine.max_steps,
        "coverage": round(engine.latest_frame.get("coverage", 0.0), 4),
        "coverage_pct": round(engine.latest_frame.get("coverage_pct", 0.0), 2),
        "total_collisions": engine.total_collisions,
        "policy_mode": engine.policy_manager.mode,
        "fps": engine.fps,
        "active_clients": len(connected_clients),
    }


@app.post("/api/simulation/start")
async def start_simulation():
    """Start or resume the continuous simulation loop."""
    if not engine:
        return {"error": "Engine not initialized"}
    await engine.start()
    return {"status": "started", "sim_status": engine.status}


@app.post("/api/simulation/pause")
async def pause_simulation():
    """Pause the continuous simulation loop."""
    if not engine:
        return {"error": "Engine not initialized"}
    engine.pause()
    return {"status": "paused", "sim_status": engine.status}


@app.post("/api/simulation/reset")
async def reset_simulation(req: Optional[ResetRequest] = None):
    """Reset the simulation environment."""
    if not engine:
        return {"error": "Engine not initialized"}
    seed = req.seed if req else None
    frame = engine.reset(seed=seed)
    await broadcast_frame(frame)
    return {"status": "reset", "frame": frame}


@app.post("/api/simulation/step")
async def step_simulation():
    """Advance the simulation by exactly one step."""
    if not engine:
        return {"error": "Engine not initialized"}
    frame = engine.step()
    await broadcast_frame(frame)
    return {"status": "stepped", "frame": frame}


@app.post("/api/simulation/config")
async def update_simulation_config(req: ConfigUpdateRequest):
    """Update runtime simulation settings (FPS, Policy mode, Fleet size)."""
    if not engine:
        return {"error": "Engine not initialized"}

    if req.fps is not None:
        engine.set_fps(req.fps)
    if req.policy is not None:
        engine.policy_manager.set_mode(req.policy)
    if req.num_drones is not None:
        engine.set_num_drones(req.num_drones)
        await broadcast_frame(engine.latest_frame)

    return {
        "status": "updated",
        "fps": engine.fps,
        "policy_mode": engine.policy_manager.mode,
        "num_drones": engine.num_drones,
    }


# ──────────────────────────────────────────────
# WebSocket Endpoint — High-Throughput Stream
# ──────────────────────────────────────────────

@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket):
    """
    WebSocket endpoint streaming drone telemetry to frontend clients.
    Supports real-time bidirectional control:
      - Client sends:
          {"action": "start"}
          {"action": "pause"}
          {"action": "reset", "seed": 42}
          {"action": "step"}
          {"action": "set_fps", "fps": 30}
          {"action": "set_policy", "policy": "trained" | "heuristic" | "random"}
          {"action": "ping"}
      - Server streams:
          {"type": "init", ...} on connect
          {"type": "state_update", ...} on each simulation frame
          {"type": "pong"} in response to ping
    """
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Client connected. Total clients: {len(connected_clients)}")

    try:
        # Send initial world config & current frame on connect
        init_payload = {
            "type": "init",
            "world": WORLD_CONFIG["world"],
            "obstacles": WORLD_CONFIG["obstacles"],
            "search_zones": WORLD_CONFIG["search_zones"],
            "num_drones": WORLD_CONFIG["drones"]["num_drones"],
            "status": engine.status if engine else "paused",
            "policy_mode": engine.policy_manager.mode if engine else "heuristic",
            "fps": engine.fps if engine else 20,
            "frame": engine.latest_frame if engine else {},
        }
        await websocket.send_text(json.dumps(init_payload))

        while True:
            data_text = await websocket.receive_text()
            try:
                cmd = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            action = cmd.get("action", "").lower()

            if action == "start":
                if engine:
                    await engine.start()
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "status": engine.status,
                        "step": engine.current_step,
                    }))

            elif action == "pause":
                if engine:
                    engine.pause()
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "status": engine.status,
                        "step": engine.current_step,
                    }))

            elif action == "reset":
                if engine:
                    seed = cmd.get("seed", None)
                    frame = engine.reset(seed=seed)
                    await broadcast_frame(frame)

            elif action == "step":
                if engine:
                    frame = engine.step()
                    await broadcast_frame(frame)

            elif action == "set_fps":
                fps = cmd.get("fps")
                if engine and isinstance(fps, (int, float)):
                    engine.set_fps(int(fps))
                    await websocket.send_text(json.dumps({
                        "type": "config_ack",
                        "fps": engine.fps,
                    }))

            elif action == "set_policy":
                pol = cmd.get("policy", "")
                if engine and pol:
                    new_mode = engine.policy_manager.set_mode(pol)
                    await websocket.send_text(json.dumps({
                        "type": "config_ack",
                        "policy_mode": new_mode,
                    }))

            elif action == "set_num_drones":
                count = cmd.get("count") or cmd.get("num_drones")
                if engine and count:
                    new_count = engine.set_num_drones(int(count))
                    await websocket.send_text(json.dumps({
                        "type": "config_ack",
                        "num_drones": new_count,
                    }))
                    await broadcast_frame(engine.latest_frame)

            elif action in ("set_curriculum_stage", "set_stage"):
                stage_id = cmd.get("stage_id") or cmd.get("stage")
                if engine and stage_id is not None:
                    new_stage = engine.set_curriculum_stage(int(stage_id))
                    await websocket.send_text(json.dumps({
                        "type": "config_ack",
                        "curriculum": new_stage,
                    }))
                    await broadcast_frame(engine.latest_frame)

            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"[WS] Connection error: {e}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
