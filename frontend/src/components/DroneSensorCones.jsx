/**
 * SwarmRL — Day 20: Drone Sensor Range & Perception Cones Component
 *
 * Renders 3D perception volumes (SENSOR_RANGE = 10.0m) around drones.
 * Provides subtle wireframe perception spheres around fleet agents, with
 * enhanced high-visibility pulsing rings on the user-selected drone.
 */

import React, { useMemo } from 'react';
import * as THREE from 'three';

const SENSOR_RADIUS = 10.0; // SENSOR_RANGE in metres

export function DroneSensorCones({ drones = [], selectedDrone = null, visible = true }) {
  if (!visible || !drones || drones.length === 0) return null;

  return (
    <group name="drone-sensor-cones">
      {drones.map((d) => {
        if (!d.alive || !d.position) return null;
        const isSelected = selectedDrone === d.id;
        // Sim coords [x, y, z] -> Three.js coords [x, z, y]
        const pos = [d.position[0], d.position[2], d.position[1]];

        return (
          <group key={d.id} position={pos}>
            {/* Subtle wireframe sensing sphere */}
            <mesh>
              <sphereGeometry args={[SENSOR_RADIUS, 16, 12]} />
              <meshBasicMaterial
                color={isSelected ? '#fbbf24' : '#00f0ff'}
                wireframe
                transparent
                opacity={isSelected ? 0.22 : 0.05}
              />
            </mesh>

            {/* Horizontal scan plane indicator */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[SENSOR_RADIUS - 0.2, SENSOR_RADIUS, 32]} />
              <meshBasicMaterial
                color={isSelected ? '#fbbf24' : '#00f0ff'}
                side={THREE.DoubleSide}
                transparent
                opacity={isSelected ? 0.35 : 0.12}
              />
            </mesh>

            {/* Directional heading ray if speed > 0.5 */}
            {d.velocity && (
              <mesh position={[d.velocity[0] * 1.5, d.velocity[2] * 1.5, d.velocity[1] * 1.5]}>
                <sphereGeometry args={[0.2, 8, 8]} />
                <meshBasicMaterial
                  color={isSelected ? '#fbbf24' : '#38bdf8'}
                  transparent
                  opacity={0.6}
                />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

export default DroneSensorCones;
