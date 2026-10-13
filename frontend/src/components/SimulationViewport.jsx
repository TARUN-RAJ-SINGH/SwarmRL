/**
 * SwarmRL — Day 12: 3D Simulation Viewport
 *
 * Full Three.js Canvas scene assembled with:
 *   - OrbitControls for camera navigation
 *   - WorldEnvironment (ground, obstacles, zones)
 *   - DroneModel instances for all active drones
 *   - Ambient + directional lighting
 *   - Fog for atmospheric depth
 *   - Stars background
 */

import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { WorldEnvironment } from './WorldEnvironment';
import { DroneModel } from './DroneModel';

function SceneContent({ worldConfig, drones, showLidar }) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.15} color="#b0c4de" />
      <directionalLight
        position={[60, 80, 40]}
        intensity={0.35}
        color="#e8f0fe"
        castShadow={false}
      />
      <directionalLight
        position={[-30, 50, -20]}
        intensity={0.15}
        color="#94a3b8"
      />

      {/* Atmospheric haze */}
      <fog attach="fog" args={['#07090e', 80, 250]} />

      {/* Stars background */}
      <Stars
        radius={200}
        depth={100}
        count={3000}
        factor={3}
        saturation={0}
        fade
        speed={0.5}
      />

      {/* World environment */}
      <WorldEnvironment worldConfig={worldConfig} />

      {/* Drone fleet */}
      {drones.map((drone) => (
        <DroneModel key={drone.id} drone={drone} showLidar={showLidar} />
      ))}
    </>
  );
}

export function SimulationViewport({ worldConfig, drones = [], showLidar = true }) {
  // Camera defaults: positioned at an isometric angle looking at center of world
  const cameraPosition = useMemo(() => {
    if (!worldConfig?.world) return [80, 60, 80];
    const w = worldConfig.world;
    const cx = (w.x_min + w.x_max) / 2;
    const cz = (w.y_min + w.y_max) / 2; // sim y -> three z
    return [cx + 70, 50, cz + 70];
  }, [worldConfig]);

  const cameraTarget = useMemo(() => {
    if (!worldConfig?.world) return [50, 10, 50];
    const w = worldConfig.world;
    return [(w.x_min + w.x_max) / 2, 10, (w.y_min + w.y_max) / 2];
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
          showLidar={showLidar}
        />
      </Suspense>

      <OrbitControls
        target={cameraTarget}
        enableDamping
        dampingFactor={0.08}
        minDistance={10}
        maxDistance={250}
        maxPolarAngle={Math.PI / 2 - 0.05}
        makeDefault
      />
    </Canvas>
  );
}

export default SimulationViewport;
