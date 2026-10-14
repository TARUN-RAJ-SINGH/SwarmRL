/**
 * SwarmRL — Day 13: 3D Volumetric Coverage Voxels
 *
 * Renders explored 3D space as a glowing particle/voxel cloud.
 * Receives newly discovered voxel coordinates from the backend and accumulates
 * them into an efficient Three.js Points geometry.
 */

import React, { useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';

const MAX_VOXELS = 1500;

export function CoverageVoxels({ newVoxels = [], step = 0, visible = true }) {
  const pointsRef = useRef(null);
  const voxelsBuffer = useRef(new Float32Array(MAX_VOXELS * 3));
  const voxelCount = useRef(0);

  // Clear voxels on simulation reset (step === 0)
  useEffect(() => {
    if (step <= 1) {
      voxelCount.current = 0;
      if (pointsRef.current) {
        pointsRef.current.geometry.setDrawRange(0, 0);
      }
    }
  }, [step]);

  // Append newly explored voxels into the buffer
  useEffect(() => {
    if (!newVoxels || newVoxels.length === 0) return;

    const buf = voxelsBuffer.current;
    let count = voxelCount.current;

    for (let i = 0; i < newVoxels.length; i++) {
      const pt = newVoxels[i];
      // sim: [x, y, z] -> three: [x, z, y]
      const px = pt[0];
      const py = pt[2]; // vertical
      const pz = pt[1];

      // Add to cyclic or capped buffer
      const idx = (count % MAX_VOXELS) * 3;
      buf[idx] = px;
      buf[idx + 1] = py;
      buf[idx + 2] = pz;
      count++;
    }

    voxelCount.current = count;

    if (pointsRef.current) {
      const geo = pointsRef.current.geometry;
      geo.attributes.position.needsUpdate = true;
      const drawCount = Math.min(count, MAX_VOXELS);
      geo.setDrawRange(0, drawCount);
    }
  }, [newVoxels]);

  // Points geometry and particle texture
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(voxelsBuffer.current, 3));
    geo.setDrawRange(0, 0);
    return geo;
  }, []);

  if (!visible) return null;

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={1.6}
        color="#10b981"
        transparent
        opacity={0.65}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
}

export default CoverageVoxels;
