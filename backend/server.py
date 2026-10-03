"""
SwarmRL — FastAPI Backend Server

Serves as the bridge between the RL inference engine and the React frontend.
Provides:
  - WebSocket endpoint for streaming drone coordinates in real time
  - REST endpoints for simulation control (start, stop, reset, configure)
  - Health check endpoint
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uvicorn

app = FastAPI(
    title="SwarmRL Backend",
    description="Real-time drone swarm simulation server",
    version="0.1.0",
)

# Allow React dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {"project": "SwarmRL", "status": "running", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/config")
async def get_config():
    """Return the current simulation configuration."""
    from env.world_config import WORLD_CONFIG
    return WORLD_CONFIG


# ──────────────────────────────────────────────
# WebSocket — Real-Time Drone Coordinate Stream
# ──────────────────────────────────────────────

connected_clients: list[WebSocket] = []


@app.websocket("/ws/simulation")
async def simulation_stream(websocket: WebSocket):
    """
    WebSocket endpoint that will stream drone positions to the frontend.
    Currently sends a placeholder heartbeat; will be replaced with
    actual simulation data once the PettingZoo environment is wired in.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Client connected. Total clients: {len(connected_clients)}")

    try:
        while True:
            # Placeholder: send a heartbeat every second
            await websocket.send_json({
                "type": "heartbeat",
                "message": "Simulation WebSocket active — environment not yet connected.",
            })
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total clients: {len(connected_clients)}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
