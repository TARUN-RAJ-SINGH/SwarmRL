/**
 * SwarmRL — Day 13: 3D Simulation Viewport & Analytics Camera System
 *
 * Full Three.js Canvas scene assembled with:
 *   - Interactive camera presets: Orbit, Tactical Top-Down, Drone Chase, Elevation
 *   - Dynamic Coverage Heatmap (2D terrain projection)
 *   - 3D Volumetric Coverage Voxels (particle cloud)
 *   - Swarm Centroid & Dispersion Marker (situational awareness)
 *   - Dynamic Drone Models with rotor discs, LiDAR ray casts, and trails
 *   - WorldEnvironment (bounding perimeter, grid, obstacles, sector completion cylinders)
 */

import React, { Suspense, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { WorldEnvironment } from './WorldEnvironment';
import { DroneModel } from './DroneModel';
import { CoverageHeatmap } from './CoverageHeatmap';
import { CoverageVoxels } from './CoverageVoxels';
import { SwarmCentroidMarker } from './SwarmCentroidMarker';
import { FlightTrails } from './FlightTrails';
import { CommunicationMesh } from './CommunicationMesh';
import { DroneSensorCones } from './DroneSensorCones';

/**
 * CameraController dynamically smoothly transitions between camera presets:
 *   - 'orbit'     : Standard interactive OrbitControls
 *   - 'topDown'   : Direct aerial 2D tactical view
 *   - 'chase'     : Smooth third-person chase camera following selected drone
 *   - 'elevation' : Horizon elevation view for obstacle clearance inspection
 */
function CameraController({ cameraMode, selectedDrone, drones, worldConfig }) {
  const { camera } = useThree();
  const controlsRef = useRef();

  useFrame((_, delta) => {
    if (!controlsRef.current) return;

    if (cameraMode === 'topDown') {
      const targetPos = new THREE.Vector3(50, 130, 50);
      const lookTarget = new THREE.Vector3(50, 0, 50);
      camera.position.lerp(targetPos, delta * 3.5);
      controlsRef.current.target.lerp(lookTarget, delta * 3.5);
      controlsRef.current.update();
    } else if (cameraMode === 'elevation') {
      const targetPos = new THREE.Vector3(50, 12, 135);
      const lookTarget = new THREE.Vector3(50, 15, 50);
      camera.position.lerp(targetPos, delta * 3.5);
      controlsRef.current.target.lerp(lookTarget, delta * 3.5);
      controlsRef.current.update();
    } else if (cameraMode === 'chase') {
      const drone = drones.find((d) => d.id === selectedDrone) || drones[0];
      if (drone && drone.position) {
        // sim [x, y, z] -> three [x, z, y]
        const px = drone.position[0];
        const py = drone.position[2]; // vertical
        const pz = drone.position[1];

        const vx = drone.velocity?.[0] || 0;
        const vz = drone.velocity?.[1] || 0;
        const speed = Math.hypot(vx, vz) || 1;
        const normVx = vx / speed;
        const normVz = vz / speed;

        const targetPos = new THREE.Vector3(
          px - normVx * 9,
          py + 4.5,
          pz - normVz * 9
        );
        const lookTarget = new THREE.Vector3(
          px + normVx * 4,
          py + 1.2,
          pz + normVz * 4
        );

        camera.position.lerp(targetPos, delta * 4);
        controlsRef.current.target.lerp(lookTarget, delta * 4);
        controlsRef.current.update();
      }
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      minDistance={6}
      maxDistance={250}
      maxPolarAngle={cameraMode === 'topDown' ? 0.05 : Math.PI / 2 - 0.05}
      makeDefault
    />
  );
}

function SceneContent({
  worldConfig,
  drones,
  commEdges,
  sectors,
  heatmapGrid,
  newVoxels,
  swarmAnalytics,
  step,
  selectedDrone,
  cameraMode,
  showLidar,
  showTrails,
  showCommMesh,
  showSensors,
  showHeatmap,
  showVoxels,
  showCentroid,
  onSelectDrone,
}) {
  return (
    <>
      {/* Balanced 3-point scene lighting */}
      <ambientLight intensity={0.18} color="#b0c4de" />
      <directionalLight
        position={[60, 80, 40]}
        intensity={0.4}
        color="#e8f0fe"
      />
      <directionalLight
        position={[-30, 50, -20]}
        intensity={0.2}
        color="#94a3b8"
      />

      {/* Atmospheric depth */}
      <fog attach="fog" args={['#07090e', 80, 260]} />

      {/* Stars backdrop */}
      <Stars
        radius={200}
        depth={100}
        count={3000}
        factor={3}
        saturation={0}
        fade
        speed={0.5}
      />

      {/* 3D World Environment with sector completion cylinders */}
      <WorldEnvironment worldConfig={worldConfig} sectors={sectors} />

      {/* Dynamic 2D Ground Coverage Heatmap */}
      <CoverageHeatmap
        heatmapGrid={heatmapGrid}
        worldConfig={worldConfig}
        visible={showHeatmap}
      />

      {/* 3D Volumetric Coverage Voxels */}
      <CoverageVoxels
        newVoxels={newVoxels}
        step={step}
        visible={showVoxels}
      />

      {/* 3D Flight Trajectory Trails */}
      <FlightTrails
        drones={drones}
        step={step}
        visible={showTrails}
      />

      {/* Dynamic GNN Inter-Agent Communication Mesh */}
      <CommunicationMesh
        commEdges={commEdges}
        drones={drones}
        visible={showCommMesh}
      />

      {/* Drone 10m Perception Sensor Cones / Bounds */}
      <DroneSensorCones
        drones={drones}
        selectedDrone={selectedDrone}
        visible={showSensors}
      />

      {/* Swarm Centroid & Spatial Dispersion Marker */}
      <SwarmCentroidMarker
        swarmAnalytics={swarmAnalytics}
        visible={showCentroid}
      />

      {/* Drone fleet */}
      {drones.map((drone) => (
        <DroneModel
          key={drone.id}
          drone={drone}
          showLidar={showLidar}
          isSelected={selectedDrone === drone.id}
          onClick={() => onSelectDrone && onSelectDrone(drone.id)}
        />
      ))}

      {/* Interactive Camera Controller */}
      <CameraController
        cameraMode={cameraMode}
        selectedDrone={selectedDrone}
        drones={drones}
        worldConfig={worldConfig}
      />
    </>
  );
}

export function SimulationViewport({
  worldConfig,
  drones = [],
  commEdges = [],
  sectors = {},
  heatmapGrid = [],
  newVoxels = [],
  swarmAnalytics = null,
  step = 0,
  selectedDrone = null,
  cameraMode = 'orbit',
  showLidar = true,
  showTrails = true,
  showCommMesh = true,
  showSensors = false,
  showHeatmap = true,
  showVoxels = true,
  showCentroid = true,
  onSelectDrone,
}) {
  const cameraPosition = useMemo(() => {
    if (!worldConfig?.world) return [80, 60, 80];
    const w = worldConfig.world;
    const cx = (w.x_min + w.x_max) / 2;
    const cz = (w.y_min + w.y_max) / 2;
    return [cx + 70, 50, cz + 70];
  }, [worldConfig]);

  return (
    <Canvas
      camera={{
        position: cameraPosition,
        fov: 50,
        near: 0.5,
        far: 500,
      }}
      gl={{
        antialias: true,
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.2,
      }}
      style={{ width: '100%', height: '100%' }}
    >
      <Suspense fallback={null}>
        <SceneContent
          worldConfig={worldConfig}
          drones={drones}
          sectors={sectors}
          heatmapGrid={heatmapGrid}
          newVoxels={newVoxels}
          swarmAnalytics={swarmAnalytics}
          step={step}
          selectedDrone={selectedDrone}
          cameraMode={cameraMode}
          commEdges={commEdges}
          showLidar={showLidar}
          showTrails={showTrails}
          showCommMesh={showCommMesh}
          showSensors={showSensors}
          showHeatmap={showHeatmap}
          showVoxels={showVoxels}
          showCentroid={showCentroid}
          onSelectDrone={onSelectDrone}
        />
      </Suspense>
    </Canvas>
  );
}

export default SimulationViewport;
