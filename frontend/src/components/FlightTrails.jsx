/**
 * SwarmRL — Day 20: 3D Flight Trajectory Trails Component
 *
 * Renders smooth dynamic fading trajectory ribbons following each drone's flight path.
 * Maintains a rolling position history and dynamically clears on simulation reset.
 */

import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { DRONE_COLORS } from './DroneModel';

const MAX_TRAIL_POINTS = 25;
const MIN_DISTANCE_DELTA = 0.15; // Minimum travel distance (m) to record a new waypoint

export function FlightTrails({ drones = [], step = 0, visible = true }) {
  // Store rolling history per drone: { [droneId]: Vector3[] }
  const historyRef = useRef({});
  const lastStepRef = useRef(step);

  // Clear trails on simulation reset or backward jump
  useEffect(() => {
    if (step === 0 || step < lastStepRef.current) {
      historyRef.current = {};
    }
    lastStepRef.current = step;
  }, [step]);

  // Update trail history whenever drones array updates
  useEffect(() => {
    if (!visible) return;

    const hist = historyRef.current;
    drones.forEach((d) => {
      if (!d.alive || !d.position) return;
      const id = d.id;
      // Convert simulation coords [x, y, z] to Three.js [x, z, y]
      const currentPos = new THREE.Vector3(d.position[0], d.position[2], d.position[1]);

      if (!hist[id]) {
        hist[id] = [currentPos];
      } else {
        const lastPos = hist[id][hist[id].length - 1];
        if (lastPos.distanceTo(currentPos) >= MIN_DISTANCE_DELTA) {
          hist[id].push(currentPos);
          if (hist[id].length > MAX_TRAIL_POINTS) {
            hist[id].shift();
          }
        }
      }
    });
  }, [drones, visible]);

  // Build line geometries for each drone
  const trailGeometries = useMemo(() => {
    if (!visible) return [];

    const hist = historyRef.current;
    const trails = [];

    drones.forEach((d) => {
      const id = d.id;
      const points = hist[id];
      if (!points || points.length < 2) return;

      const color = DRONE_COLORS[id % DRONE_COLORS.length];
      const geo = new THREE.BufferGeometry().setFromPoints(points);

      trails.push({
        id,
        geometry: geo,
        color,
      });
    });

    return trails;
  }, [drones, visible, step]);

  if (!visible || trailGeometries.length === 0) return null;

  return (
    <group name="flight-trails">
      {trailGeometries.map((t) => (
        <line key={t.id} geometry={t.geometry}>
          <lineBasicMaterial
            color={t.color}
            transparent
            opacity={0.55}
            linewidth={2}
            blending={THREE.AdditiveBlending}
          />
        </line>
      ))}
    </group>
  );
}

export default FlightTrails;
