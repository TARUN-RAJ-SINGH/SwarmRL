import React from 'react';
import { useSimulation } from './hooks/useSimulation';
import './App.css';

export function App() {
  const {
    connected,
    simState,
    worldConfig,
    isRunning,
    start,
    pause,
    step,
    reset,
    setPolicy,
    setFps,
  } = useSimulation();

  const drones = simState.drones || [];
  const coveragePercent = simState.coverage_pct || 0;
  const aliveDrones = drones.filter((d) => d.alive).length;

  return (
    <div className="mission-control">
      {/* ── Top Navigation Bar ── */}
      <header className="top-bar">
        <div className="brand">
          <div className="radar-icon">
            <span className="radar-ring"></span>
            <span className="radar-blip"></span>
          </div>
          <div className="brand-text">
            <div className="brand-title">SwarmRL</div>
            <div className="brand-subtitle">Autonomous Drone Swarm Telemetry</div>
          </div>
        </div>

        <div className="header-meta">
          <div className={`status-pill ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            <span>{connected ? 'WS LIVE (8000)' : 'OFFLINE'}</span>
          </div>

          <div className={`sim-badge status-${simState.status}`}>
            {simState.status.toUpperCase()}
          </div>
        </div>
      </header>

      {/* ── Top Metrics Strip ── */}
      <section className="metrics-bar">
        <div className="metric-card">
          <div className="metric-label">GLOBAL COVERAGE</div>
          <div className="metric-value">
            <span className="accent-val">{coveragePercent.toFixed(1)}</span>
            <span className="unit">%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(coveragePercent, 100)}%` }}
            ></div>
          </div>
          <div className="metric-sub">Target: 85.0%</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">ACTIVE SWARM</div>
          <div className="metric-value">
            <span className="accent-val">{aliveDrones}</span>
            <span className="unit">/ {worldConfig?.num_drones || 10}</span>
          </div>
          <div className="swarm-dots">
            {drones.map((d) => (
              <span
                key={d.id}
                className={`drone-dot ${d.alive ? 'active' : 'inactive'}`}
                title={`Drone ${d.id}: ${d.alive ? 'Online' : 'Lost'}`}
              ></span>
            ))}
          </div>
          <div className="metric-sub">Clustered Basecamp</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">SIMULATION STEP</div>
          <div className="metric-value">
            <span className="accent-val">{simState.step}</span>
            <span className="unit">/ {simState.max_steps}</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill step-fill"
              style={{
                width: `${Math.min((simState.step / simState.max_steps) * 100, 100)}%`,
              }}
            ></div>
          </div>
          <div className="metric-sub">{simState.fps} Hz Stream Rate</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">COLLISIONS</div>
          <div className="metric-value">
            <span className={`accent-val ${simState.total_collisions > 0 ? 'alert' : ''}`}>
              {simState.total_collisions}
            </span>
          </div>
          <div className="metric-sub">Penalties Active (-2.0)</div>
        </div>
      </section>

      {/* ── Control Console Toolbar ── */}
      <section className="toolbar">
        <div className="control-group">
          <button
            id="btn-play-pause"
            className={`btn-control ${isRunning ? 'btn-pause' : 'btn-play'}`}
            onClick={isRunning ? pause : start}
            disabled={!connected}
          >
            {isRunning ? '⏸ Pause Sim' : '▶ Start Swarm'}
          </button>

          <button
            id="btn-step"
            className="btn-control btn-secondary"
            onClick={step}
            disabled={!connected || isRunning}
            title="Advance 1 simulation step"
          >
            ⏭ Step Forward
          </button>

          <button
            id="btn-reset"
            className="btn-control btn-secondary"
            onClick={() => reset(42)}
            disabled={!connected}
            title="Reset swarm to basecamp"
          >
            ↺ Reset Swarm
          </button>
        </div>

        <div className="control-group">
          <label className="select-label">Policy Model:</label>
          <div className="segmented-control">
            {['trained', 'heuristic', 'random'].map((mode) => (
              <button
                key={mode}
                className={`segment-btn ${simState.policy_mode === mode ? 'active' : ''}`}
                onClick={() => setPolicy(mode)}
                disabled={!connected}
              >
                {mode === 'trained' ? 'RL MAPPO' : mode === 'heuristic' ? 'APF Heuristic' : 'Random'}
              </button>
            ))}
          </div>

          <label className="select-label">Speed:</label>
          <div className="segmented-control">
            {[10, 20, 30, 60].map((rate) => (
              <button
                key={rate}
                className={`segment-btn ${simState.fps === rate ? 'active' : ''}`}
                onClick={() => setFps(rate)}
                disabled={!connected}
              >
                {rate} FPS
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Main Workspace ── */}
      <main className="workspace-grid">
        {/* Viewport Card */}
        <div className="viewport-container">
          <div className="panel-header">
            <span>3D TACTICAL DISASTER ZONE (100m × 100m × 50m)</span>
            <span className="panel-tag">Day 12 Viewport Mount Ready</span>
          </div>

          <div className="viewport-stage">
            <div className="grid-overlay"></div>
            <div className="radar-sweep"></div>

            <div className="viewport-center-info">
              <div className="viewport-icon">🛸</div>
              <h3>3D Three.js Viewport Integration</h3>
              <p>
                WebSocket telemetry streaming active at {simState.fps} Hz.
                Positions, velocities, LiDAR rays, and collision meshes ready for Day 12 Canvas mounting.
              </p>
              <div className="specs-list">
                <div className="spec-badge">Bounds: 100×100×50m</div>
                <div className="spec-badge">Obstacles: 5 Built</div>
                <div className="spec-badge">Voxel Grid: 62,500 Cells</div>
                <div className="spec-badge">LiDAR: 10 Rays/Drone</div>
              </div>
            </div>

            <div className="viewport-coords-overlay">
              <div>ORIGIN: [0, 0, 0]</div>
              <div>EXTENT: [100, 100, 50]</div>
              <div>CLUSTERING: Basecamp [10, 10, 5]</div>
            </div>
          </div>
        </div>

        {/* Live Swarm Telemetry Panel */}
        <div className="telemetry-panel">
          <div className="panel-header">
            <span>LIVE SWARM TELEMETRY</span>
            <span className="panel-count">{drones.length} DRONES</span>
          </div>

          <div className="telemetry-table-wrap">
            <table className="telemetry-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Position [x, y, z]</th>
                  <th>Velocity [vx, vy, vz]</th>
                  <th>Speed</th>
                  <th>Voxels</th>
                </tr>
              </thead>
              <tbody>
                {drones.map((d) => (
                  <tr key={d.id} className={d.alive ? '' : 'dead-row'}>
                    <td className="drone-id-col">
                      <span className="agent-badge">#{d.id}</span>
                    </td>
                    <td className="coord-col">
                      [{d.position.map((v) => v.toFixed(1)).join(', ')}]
                    </td>
                    <td className="coord-col">
                      [{d.velocity.map((v) => v.toFixed(1)).join(', ')}]
                    </td>
                    <td>{d.speed.toFixed(2)} m/s</td>
                    <td className="cells-col">+{d.cells_explored}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
