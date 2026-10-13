/**
 * SwarmRL — 3D Drone Model Component
 *
 * Renders a single drone as a glowing sphere with:
 *   - Animated rotor disc
 *   - Velocity-aligned orientation
 *   - Positional glow point light
 *   - LiDAR ray beams (thin cyan lines from drone to hit points)
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Colour palette per drone ID (10 distinct hues)
const DRONE_COLORS = [
  '#00f0ff', // cyan
  '#10b981', // emerald
  '#f59e0b', // amber
  '#f43f5e', // rose
  '#8b5cf6', // violet
  '#06b6d4', // teal
  '#ec4899', // pink
  '#84cc16', // lime
  '#f97316', // orange
  '#6366f1', // indigo
];

export function DroneModel({ drone, showLidar = true }) {
  const groupRef = useRef();
  const rotorRef = useRef();
  const glowRef = useRef();

  const color = DRONE_COLORS[drone.id % DRONE_COLORS.length];
  const pos = drone.position;
  const speed = drone.speed || 0;

  // Animate rotor spin and glow pulse
  useFrame((_, delta) => {
    if (rotorRef.current) {
      rotorRef.current.rotation.y += (4 + speed * 2) * delta;
    }
    if (glowRef.current) {
      glowRef.current.intensity = 0.4 + Math.sin(Date.now() * 0.003 + drone.id) * 0.15;
    }
  });

  // LiDAR beam geometry: lines from drone position to each hit point
  const lidarGeometry = useMemo(() => {
    if (!showLidar || !drone.lidar_hits || drone.lidar_hits.length === 0) return null;
    const points = [];
    for (const hit of drone.lidar_hits) {
      points.push(new THREE.Vector3(pos[0], pos[2], pos[1])); // drone origin (swizzle y/z)
      points.push(new THREE.Vector3(hit[0], hit[2], hit[1])); // hit endpoint
    }
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return geo;
  }, [pos[0], pos[1], pos[2], showLidar, drone.lidar_hits]);

  return (
    <group ref={groupRef} position={[pos[0], pos[2], pos[1]]}>
      {/* Main body sphere */}
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.6}
          metalness={0.7}
          roughness={0.2}
          transparent
          opacity={drone.alive ? 0.95 : 0.3}
        />
      </mesh>

      {/* Inner core glow */}
      <mesh>
        <sphereGeometry args={[0.25, 12, 12]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
      </mesh>

      {/* Rotor disc */}
      <mesh ref={rotorRef} position={[0, 0.55, 0]} rotation={[0, 0, 0]}>
        <cylinderGeometry args={[0.7, 0.7, 0.04, 6]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.3}
          metalness={0.8}
          roughness={0.1}
        />
      </mesh>

      {/* Point light glow */}
      <pointLight ref={glowRef} color={color} intensity={0.4} distance={8} decay={2} />

      {/* Drone ID label - small sphere indicator above */}
      <mesh position={[0, 1.1, 0]}>
        <sphereGeometry args={[0.12, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>

      {/* LiDAR beams rendered at world coordinates, so offset back */}
      {lidarGeometry && (
        <lineSegments
          geometry={lidarGeometry}
          position={[-pos[0], -pos[2], -pos[1]]} // undo group position so world coords work
        >
          <lineBasicMaterial
            color={color}
            transparent
            opacity={0.25}
            linewidth={1}
          />
        </lineSegments>
      )}
    </group>
  );
}

export default DroneModel;
