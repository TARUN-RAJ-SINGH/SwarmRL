/**
 * SwarmRL — React Hook for Real-Time Swarm Simulation
 * 
 * Provides reactive access to live telemetry, connection state,
 * world configuration, and simulation control handlers.
 */

import { useState, useEffect, useCallback } from 'react';
import { simulationSocket } from '../services/simulationSocket';

export function useSimulation() {
  const [connected, setConnected] = useState(false);
  const [worldConfig, setWorldConfig] = useState(null);
  const [simState, setSimState] = useState({
    step: 0,
    max_steps: 1000,
    coverage: 0,
    coverage_pct: 0,
    new_cells: 0,
    total_collisions: 0,
    status: 'paused',
    policy_mode: 'trained',
    fps: 20,
    drones: [],
    events: {
      drone_drone_collisions: [],
      drone_obstacle_collisions: [],
    },
    is_terminated: false,
    is_truncated: false,
  });

  useEffect(() => {
    // 1. Initial connect
    simulationSocket.connect();

    // 2. Event Handlers
    const unsubConnect = simulationSocket.on('connect', () => {
      setConnected(true);
    });

    const unsubDisconnect = simulationSocket.on('disconnect', () => {
      setConnected(false);
    });

    const unsubInit = simulationSocket.on('init', (data) => {
      setWorldConfig({
        world: data.world,
        obstacles: data.obstacles,
        search_zones: data.search_zones,
        num_drones: data.num_drones,
      });
      if (data.frame && Object.keys(data.frame).length > 0) {
        setSimState((prev) => ({ ...prev, ...data.frame }));
      }
    });

    const unsubUpdate = simulationSocket.on('state_update', (data) => {
      setSimState((prev) => ({
        ...prev,
        ...data,
      }));
    });

    const unsubStatus = simulationSocket.on('status', (data) => {
      setSimState((prev) => ({
        ...prev,
        status: data.status,
        step: data.step ?? prev.step,
      }));
    });

    const unsubAck = simulationSocket.on('config_ack', (data) => {
      setSimState((prev) => ({
        ...prev,
        policy_mode: data.policy_mode ?? prev.policy_mode,
        fps: data.fps ?? prev.fps,
        num_drones: data.num_drones ?? prev.num_drones,
      }));
    });

    return () => {
      unsubConnect();
      unsubDisconnect();
      unsubInit();
      unsubUpdate();
      unsubStatus();
      unsubAck();
    };
  }, []);

  // Control handlers
  const start = useCallback(() => simulationSocket.start(), []);
  const pause = useCallback(() => simulationSocket.pause(), []);
  const step = useCallback(() => simulationSocket.step(), []);
  const reset = useCallback((seed = null) => simulationSocket.reset(seed), []);
  const setPolicy = useCallback((policy) => simulationSocket.setPolicy(policy), []);
  const setFps = useCallback((fps) => simulationSocket.setFps(fps), []);
  const setNumDrones = useCallback((count) => simulationSocket.setNumDrones(count), []);

  return {
    connected,
    worldConfig,
    simState,
    isRunning: simState.status === 'running',
    start,
    pause,
    step,
    reset,
    setPolicy,
    setFps,
    setNumDrones,
  };
}

export default useSimulation;
