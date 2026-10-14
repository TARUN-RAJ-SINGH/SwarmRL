/**
 * SwarmRL — Day 13: Swarm Centroid & Dispersion Marker
 *
 * Visualizes the swarm's center of mass (centroid) and spatial dispersion radius:
 *   - Holographic rotating octahedron beacon at centroid
 *   - Vertical tactical laser projection line to ground
 *   - Ground circle representing the fleet's dispersion envelope
 */

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function SwarmCentroidMarker({ swarmAnalytics, visible = true }) {
  const meshRef = useRef();

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 1.2;
      meshRef.current.rotation.x += delta * 0.4;
    }
  });

  if (!visible || !swarmAnalytics || !swarmAnalytics.centroid) return null;

  const [cx, cz_sim, cy_sim] = swarmAnalytics.centroid;
  // Sim coords: [x, y, z] -> Three coords: [x, z, y]
  const px = cx;
  const py = cy_sim; // height in three.js
  const pz = cz_sim;

  const dispersion = Math.max(swarmAnalytics.dispersion_radius || 2, 0.5);

  return (
    <group position={[px, 0, pz]}>
      {/* Ground dispersion perimeter ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
        <ringGeometry args={[dispersion - 0.25, dispersion, 48]} />
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Vertical projection line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, py, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#38bdf8" transparent opacity={0.4} />
      </line>

      {/* Centroid beacon diamond */}
      <group position={[0, py, 0]}>
        <mesh ref={meshRef}>
          <octahedronGeometry args={[0.9, 0]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#0284c7"
            emissiveIntensity={0.8}
            wireframe
          />
        </mesh>
        <pointLight color="#38bdf8" intensity={1.5} distance={15} />
      </group>
    </group>
  );
}

export default SwarmCentroidMarker;
