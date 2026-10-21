/**
 * SwarmRL — Day 20: GNN Inter-Agent Communication Mesh Component
 *
 * Renders dynamic glowing communication links connecting drones that are within
 * mutual communication range (COMM_RANGE = 20.0m). Assembles all active links
 * into a single unified LineSegments buffer geometry for optimal WebGL performance.
 */

import React, { useMemo } from 'react';
import * as THREE from 'three';

export function CommunicationMesh({ commEdges = [], drones = [], visible = true }) {
  const meshData = useMemo(() => {
    if (!visible || !commEdges || commEdges.length === 0 || !drones || drones.length === 0) {
      return null;
    }

    // Build fast drone position lookup map: { [id]: Vector3 }
    const posMap = new Map();
    for (const d of drones) {
      if (d.alive && d.position) {
        // Sim coords [x, y, z] -> Three.js coords [x, z, y]
        posMap.set(d.id, new THREE.Vector3(d.position[0], d.position[2], d.position[1]));
      }
    }

    const points = [];
    const colors = [];
    const baseColor = new THREE.Color('#00f0ff'); // Neon cyan
    const closeColor = new THREE.Color('#10b981'); // Emerald green for close formation

    for (const edge of commEdges) {
      const p1 = posMap.get(edge.source);
      const p2 = posMap.get(edge.target);
      if (!p1 || !p2) continue;

      points.push(p1);
      points.push(p2);

      // Color lerp based on distance: close drones (5m-10m) glow emerald/cyan
      const ratio = Math.min(Math.max((edge.distance - 2.0) / 18.0, 0.0), 1.0);
      const edgeCol = new THREE.Color().lerpColors(closeColor, baseColor, ratio);

      colors.push(edgeCol.r, edgeCol.g, edgeCol.b);
      colors.push(edgeCol.r, edgeCol.g, edgeCol.b);
    }

    if (points.length === 0) return null;

    const geo = new THREE.BufferGeometry().setFromPoints(points);
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return {
      geometry: geo,
      edgeCount: points.length / 2,
    };
  }, [commEdges, drones, visible]);

  if (!visible || !meshData) return null;

  return (
    <lineSegments geometry={meshData.geometry} name="comm-mesh">
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={0.4}
        blending={THREE.AdditiveBlending}
        linewidth={1.5}
      />
    </lineSegments>
  );
}

export default CommunicationMesh;
