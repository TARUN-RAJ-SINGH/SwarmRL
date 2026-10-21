import React, { useState } from 'react';
import { useSimulation } from './hooks/useSimulation';
import { SimulationViewport } from './components/SimulationViewport';
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
    setNumDrones,
    setCurriculumStage,
  } = useSimulation();

  // Day 20 Camera, Trails, Mesh & Layer States
  const [cameraMode, setCameraMode] = useState('orbit'); // 'orbit' | 'topDown' | 'chase' | 'elevation'
  const [showTrails, setShowTrails] = useState(true);
  const [showCommMesh, setShowCommMesh] = useState(true);
  const [showSensors, setShowSensors] = useState(false);
  const [showLidar, setShowLidar] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showVoxels, setShowVoxels] = useState(true);
  const [showCentroid, setShowCentroid] = useState(true);
  const [selectedDrone, setSelectedDrone] = useState(null);

  const drones = simState.drones || [];
  const commEdges = simState.comm_edges || [];
  const curriculum = simState.curriculum || {
    stage_id: 1,
    stage_name: 'Dispersion & Safety',
    target_coverage: 15.0,
    description: 'Cluster dispersal with collision avoidance',
  };
  const coveragePercent = simState.coverage_pct || 0;
  const aliveDrones = drones.filter((d) => d.alive).length;
  const sectors = simState.sectors || {};
  const swarmAnalytics = simState.swarm_analytics || {
    centroid: [10, 10, 5],
    dispersion_radius: 0,
    min_distance: 0,
    avg_distance: 0,
    avg_speed: 0,
    comm_links: 0,
    avg_neighbors: 0,
    network_density: 0,
  };

  const selectedDroneData = drones.find((d) => d.id === selectedDrone);

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
            <div className="brand-subtitle">Autonomous Drone Swarm Telemetry & Analytics</div>
          </div>
        </div>

        <div className="header-meta">
          {/* Day 20: Curriculum Stage Badge */}
          <div className="curriculum-badge">
            <span className="curriculum-icon">🎯</span>
            <div className="curriculum-content">
              <span className="curriculum-stage-name">
                STAGE {curriculum.stage_id}: {curriculum.stage_name.toUpperCase()}
              </span>
              <span className="curriculum-target">Target: {curriculum.target_coverage}% Cov</span>
            </div>
          </div>

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
          <div className="metric-sub">Curriculum Goal: {curriculum.target_coverage}%</div>
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
                className={`drone-dot ${d.alive ? 'active' : 'inactive'} ${selectedDrone === d.id ? 'dot-selected' : ''}`}
                title={`Drone ${d.id}: ${d.alive ? 'Online' : 'Lost'}`}
                onClick={() => setSelectedDrone(selectedDrone === d.id ? null : d.id)}
              ></span>
            ))}
          </div>
          <div className="metric-sub">
            {selectedDrone !== null ? `Drone #${selectedDrone} Locked` : 'Basecamp Fleet'}
          </div>
        </div>

        {/* Day 20: GNN Communication Mesh Metric */}
        <div className="metric-card">
          <div className="metric-label">COMMUNICATION MESH</div>
          <div className="metric-value">
            <span className="accent-val">{swarmAnalytics.comm_links || commEdges.length}</span>
            <span className="unit">Links</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill mesh-fill"
              style={{ width: `${Math.min(((swarmAnalytics.network_density || 0) * 100), 100)}%` }}
            ></div>
          </div>
          <div className="metric-sub">
            {((swarmAnalytics.network_density || 0) * 100).toFixed(0)}% Density • {swarmAnalytics.avg_neighbors?.toFixed(1) || '0.0'} Peers/Drone
          </div>
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
          <label className="select-label">Policy:</label>
          <div className="segmented-control">
            {[
              { id: 'trained', label: '⚡ MAPPO (Day 19)' },
              { id: 'heuristic', label: '🛡️ Heuristic APF' },
              { id: 'random', label: '🎲 Random' },
            ].map((p) => (
              <button
                key={p.id}
                className={`segment-btn ${simState.policy_mode === p.id ? 'active' : ''}`}
                onClick={() => setPolicy(p.id)}
                disabled={!connected}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Day 20: Curriculum Stage Selector */}
          <label className="select-label">Curriculum:</label>
          <div className="segmented-control">
            {[1, 2, 3].map((sId) => (
              <button
                key={sId}
                className={`segment-btn ${curriculum.stage_id === sId ? 'active' : ''}`}
                onClick={() => setCurriculumStage && setCurriculumStage(sId)}
                disabled={!connected}
                title={`Switch to Curriculum Stage ${sId}`}
              >
                Stage {sId}
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

          <label className="select-label">Fleet:</label>
          <div className="segmented-control">
            {[10, 25, 50].map((count) => (
              <button
                key={count}
                className={`segment-btn ${(simState.num_drones || drones.length || 10) === count ? 'active' : ''}`}
                onClick={() => setNumDrones(count)}
                disabled={!connected}
                title={`Scale swarm fleet size to ${count} drones`}
              >
                {count} Agents
              </button>
            ))}
          </div>
        </div>

        {/* Day 20: 3D Visualization Layer Toggles */}
        <div className="control-group layer-toggles">
          <button
            className={`btn-control btn-toggle ${showTrails ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowTrails(!showTrails)}
            title="Toggle 3D drone trajectory trails"
          >
            {showTrails ? '◉ Trails' : '○ Trails'}
          </button>
          <button
            className={`btn-control btn-toggle ${showCommMesh ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowCommMesh(!showCommMesh)}
            title="Toggle dynamic GNN inter-agent communication mesh"
          >
            {showCommMesh ? '◉ Comm Mesh' : '○ Comm Mesh'}
          </button>
          <button
            className={`btn-control btn-toggle ${showSensors ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowSensors(!showSensors)}
            title="Toggle 10m perception bounds & sensor cones"
          >
            {showSensors ? '◉ Sensors' : '○ Sensors'}
          </button>
          <button
            className={`btn-control btn-toggle ${showLidar ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowLidar(!showLidar)}
            title="Toggle LiDAR laser beams"
          >
            {showLidar ? '◉ LiDAR' : '○ LiDAR'}
          </button>
          <button
            className={`btn-control btn-toggle ${showHeatmap ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowHeatmap(!showHeatmap)}
            title="Toggle 2D ground coverage heatmap"
          >
            {showHeatmap ? '◉ Heatmap' : '○ Heatmap'}
          </button>
          <button
            className={`btn-control btn-toggle ${showVoxels ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowVoxels(!showVoxels)}
            title="Toggle 3D volumetric voxels"
          >
            {showVoxels ? '◉ 3D Voxels' : '○ 3D Voxels'}
          </button>
          <button
            className={`btn-control btn-toggle ${showCentroid ? 'btn-toggle-on' : ''}`}
            onClick={() => setShowCentroid(!showCentroid)}
            title="Toggle swarm centroid beacon & spread"
          >
            {showCentroid ? '◉ Centroid' : '○ Centroid'}
          </button>
        </div>
      </section>

      {/* ── Main Workspace ── */}
      <main className="workspace-grid">
        {/* 3D Viewport Column */}
        <div className="viewport-container">
          <div className="panel-header">
            <div className="header-left">
              <span>3D TACTICAL DISASTER ZONE (100m × 100m × 50m)</span>
              <span className="panel-tag">
                {drones.length} DRONES • {simState.policy_mode.toUpperCase()} • STAGE {curriculum.stage_id}
              </span>
            </div>

            {/* Camera Presets Selector */}
            <div className="camera-controls">
              <span className="camera-label">CAMERA:</span>
              <div className="segmented-control">
                {[
                  { id: 'orbit', label: '🌐 Orbit' },
                  { id: 'topDown', label: '🎯 Top-Down' },
                  { id: 'chase', label: '🚁 Chase' },
                  { id: 'elevation', label: '📐 Elevation' },
                ].map((cam) => (
                  <button
                    key={cam.id}
                    className={`segment-btn ${cameraMode === cam.id ? 'active' : ''}`}
                    onClick={() => setCameraMode(cam.id)}
                    title={`Switch to ${cam.label} view`}
                  >
                    {cam.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="viewport-canvas-wrap">
            <SimulationViewport
              worldConfig={worldConfig}
              drones={drones}
              commEdges={commEdges}
              sectors={sectors}
              heatmapGrid={simState.heatmap_grid}
              newVoxels={simState.new_voxels}
              swarmAnalytics={swarmAnalytics}
              step={simState.step}
              selectedDrone={selectedDrone}
              cameraMode={cameraMode}
              showLidar={showLidar}
              showTrails={showTrails}
              showCommMesh={showCommMesh}
              showSensors={showSensors}
              showHeatmap={showHeatmap}
              showVoxels={showVoxels}
              showCentroid={showCentroid}
              onSelectDrone={setSelectedDrone}
            />

            {/* Floating HUD overlay on top of canvas */}
            <div className="viewport-hud">
              <div className="hud-item">
                <span className="hud-label">CAM</span> {cameraMode.toUpperCase()}
              </div>
              <div className="hud-item">
                <span className="hud-label">COV</span> {coveragePercent.toFixed(1)}%
              </div>
              <div className="hud-item">
                <span className="hud-label">STAGE</span> S{curriculum.stage_id} ({curriculum.target_coverage}%)
              </div>
              <div className="hud-item">
                <span className="hud-label">COMM</span> {commEdges.length} LINKS
              </div>
              <div className="hud-item">
                <span className="hud-label">STEP</span> {simState.step}/{simState.max_steps}
              </div>
              {selectedDroneData && (
                <div className="hud-item hud-selected">
                  <span className="hud-label">DRONE #{selectedDroneData.id}</span>
                  <span>[{selectedDroneData.position.map((v) => v.toFixed(1)).join(', ')}]</span>
                  <span>{selectedDroneData.speed.toFixed(1)} m/s</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Sidebar: Sector Completion & Swarm Analytics & Telemetry */}
        <aside className="telemetry-panel">
          {/* Day 13: Search Sector Completion Meters */}
          <div className="analytics-card">
            <div className="panel-subhead">SEARCH SECTORS COMPLETION</div>
            <div className="sectors-grid">
              {Object.entries(sectors).map(([id, sec]) => {
                const pct = sec.coverage_pct || 0;
                const isCleared = pct >= 85.0;
                return (
                  <div key={id} className={`sector-item ${isCleared ? 'sector-cleared' : ''}`}>
                    <div className="sector-header">
                      <span className="sector-name">{sec.name}</span>
                      <span className={`priority-badge priority-${sec.priority}`}>
                        {sec.priority.toUpperCase()}
                      </span>
                    </div>
                    <div className="sector-value-row">
                      <span className="sector-val">{pct.toFixed(1)}%</span>
                      <span className="sector-target">Target: {sec.target_pct}%</span>
                    </div>
                    <div className="progress-track sector-track">
                      <div
                        className={`progress-fill ${sec.priority === 'high' ? 'fill-high' : 'fill-med'}`}
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Day 20: Swarm Formation & Network Topology Analytics */}
          <div className="analytics-card">
            <div className="panel-subhead">SWARM FORMATION & GNN TOPOLOGY</div>
            <div className="analytics-grid">
              <div className="analytic-stat">
                <div className="stat-label">DISPERSION RADIUS</div>
                <div className="stat-val">
                  {swarmAnalytics.dispersion_radius?.toFixed(1) || '0.0'}
                  <span className="unit">m</span>
                </div>
                <div className="stat-desc">Swarm Spatial Spread</div>
              </div>

              <div className="analytic-stat">
                <div className="stat-label">MIN SEPARATION</div>
                <div className="stat-val">
                  <span className={swarmAnalytics.min_distance < 2.0 ? 'alert' : ''}>
                    {swarmAnalytics.min_distance?.toFixed(1) || '0.0'}
                  </span>
                  <span className="unit">m</span>
                </div>
                <div className="stat-desc">Safety Clearance</div>
              </div>

              <div className="analytic-stat">
                <div className="stat-label">COMM MESH LINKS</div>
                <div className="stat-val">
                  <span className="accent-val">{swarmAnalytics.comm_links || commEdges.length}</span>
                  <span className="unit">links</span>
                </div>
                <div className="stat-desc">
                  {((swarmAnalytics.network_density || 0) * 100).toFixed(0)}% Network Density
                </div>
              </div>

              <div className="analytic-stat">
                <div className="stat-label">AVG PEERS / DRONE</div>
                <div className="stat-val">
                  {swarmAnalytics.avg_neighbors?.toFixed(1) || '0.0'}
                  <span className="unit">nodes</span>
                </div>
                <div className="stat-desc">20m Comm Radius</div>
              </div>

              <div className="analytic-stat">
                <div className="stat-label">SWARM CENTROID</div>
                <div className="stat-val-sm font-mono">
                  [{swarmAnalytics.centroid ? swarmAnalytics.centroid.map((v) => v.toFixed(1)).join(', ') : '0, 0, 0'}]
                </div>
                <div className="stat-desc">Center of Mass</div>
              </div>

              <div className="analytic-stat">
                <div className="stat-label">FLEET AVG SPEED</div>
                <div className="stat-val">
                  {swarmAnalytics.avg_speed?.toFixed(1) || '0.0'}
                  <span className="unit">m/s</span>
                </div>
                <div className="stat-desc">Mean Flight Velocity</div>
              </div>
            </div>
          </div>

          {/* Live Swarm Telemetry Table */}
          <div className="table-section">
            <div className="panel-subhead">
              <span>ACTIVE FLEET TELEMETRY</span>
              <span className="panel-count">{drones.length} AGENTS</span>
            </div>

            <div className="telemetry-table-wrap">
              <table className="telemetry-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Position [x, y, z]</th>
                    <th>Speed</th>
                    <th>Voxels</th>
                  </tr>
                </thead>
                <tbody>
                  {drones.map((d) => (
                    <tr
                      key={d.id}
                      className={`${d.alive ? '' : 'dead-row'} ${selectedDrone === d.id ? 'selected-row' : ''}`}
                      onClick={() => setSelectedDrone(selectedDrone === d.id ? null : d.id)}
                    >
                      <td className="drone-id-col">
                        <span className="agent-badge">#{d.id}</span>
                      </td>
                      <td className="coord-col">
                        [{d.position.map((v) => v.toFixed(1)).join(', ')}]
                      </td>
                      <td>{d.speed.toFixed(2)} m/s</td>
                      <td className="cells-col">+{d.cells_explored}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
