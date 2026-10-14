/**
 * SwarmRL — 3D World Environment
 *
 * Renders the 3D disaster zone:
 *   - Wireframe bounding box (100 × 100 × 50m)
 *   - Ground plane with grid
 *   - Obstacle meshes (buildings, rubble, walls) with translucent glow
 *   - Search zone indicators (flat cylinders on ground)
 *   - Basecamp spawn marker
 */

import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Grid } from '@react-three/drei';

/**
 * Coordinate mapping: simulation uses (x, y, z) where y is forward, z is up.
 * Three.js uses (x, y, z) where y is up. So we swizzle: sim(x, y, z) -> three(x, z, y).
 */

// Obstacle type color map
const OBS_COLORS = {
  building: { base: '#1e3a5f', emissive: '#0a2240', edge: '#2563eb' },
  rubble:   { base: '#4a3728', emissive: '#2d1f14', edge: '#92400e' },
  wall:     { base: '#3b2d4a', emissive: '#1e1530', edge: '#7c3aed' },
};

function getObsStyle(id) {
  if (id.startsWith('building')) return OBS_COLORS.building;
  if (id.startsWith('rubble')) return OBS_COLORS.rubble;
  if (id.startsWith('wall')) return OBS_COLORS.wall;
  return OBS_COLORS.building;
}

export function WorldEnvironment({ worldConfig, sectors = {} }) {
  if (!worldConfig) return null;

  const world = worldConfig.world;
  const obstacles = worldConfig.obstacles || [];
  const searchZones = worldConfig.search_zones || [];

  // World dimensions
  const wx = world.x_max - world.x_min;
  const wy = world.y_max - world.y_min;
  const wz = world.z_max - world.z_min;

  // Center of bounding box in Three.js coords (swap y/z)
  const cx = world.x_min + wx / 2;
  const cy = world.z_min + wz / 2; // z_sim -> y_three
  const cz = world.y_min + wy / 2; // y_sim -> z_three

  // Wireframe bounding box edges
  const boundaryEdges = useMemo(() => {
    const geo = new THREE.BoxGeometry(wx, wz, wy);
    const edges = new THREE.EdgesGeometry(geo);
    return edges;
  }, [wx, wy, wz]);

  return (
    <group>
      {/* Ground grid (at y=0 in Three.js = z=0 in sim) */}
      <Grid
        position={[cx, 0.01, cz]}
        args={[wx, wy]}
        cellSize={2}
        cellThickness={0.4}
        cellColor="#0d2847"
        sectionSize={10}
        sectionThickness={0.8}
        sectionColor="#1e3a5f"
        fadeDistance={200}
        fadeStrength={1}
        infiniteGrid={false}
      />

      {/* Ground plane (dark semi-transparent) */}
      <mesh position={[cx, 0, cz]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[wx, wy]} />
        <meshStandardMaterial
          color="#060a12"
          transparent
          opacity={0.85}
          metalness={0.1}
          roughness={0.95}
        />
      </mesh>

      {/* Bounding box wireframe */}
      <lineSegments geometry={boundaryEdges} position={[cx, cy, cz]}>
        <lineBasicMaterial color="#00f0ff" transparent opacity={0.2} />
      </lineSegments>

      {/* Corner axis markers */}
      <group position={[world.x_min, 0, world.y_min]}>
        {/* X axis - red */}
        <mesh position={[5, 0.05, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <cylinderGeometry args={[0.08, 0.08, 10, 6]} />
          <meshBasicMaterial color="#ef4444" transparent opacity={0.6} />
        </mesh>
        {/* Z axis (sim Y) - green */}
        <mesh position={[0, 0.05, 5]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 10, 6]} />
          <meshBasicMaterial color="#22c55e" transparent opacity={0.6} />
        </mesh>
        {/* Y axis (sim Z / up) - blue */}
        <mesh position={[0, 5, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 10, 6]} />
          <meshBasicMaterial color="#3b82f6" transparent opacity={0.6} />
        </mesh>
      </group>

      {/* Obstacles */}
      {obstacles.map((obs) => {
        const style = getObsStyle(obs.id);
        const sx = obs.size[0];
        const sy = obs.size[1];
        const sz = obs.size[2];
        // Position: obstacle origin is bottom-corner in sim coords
        const px = obs.position[0] + sx / 2;
        const py = obs.position[2] + sz / 2; // z_sim -> y_three (vertical)
        const pz = obs.position[1] + sy / 2; // y_sim -> z_three

        const boxGeo = new THREE.BoxGeometry(sx, sz, sy);
        const edgeGeo = new THREE.EdgesGeometry(boxGeo);

        return (
          <group key={obs.id} position={[px, py, pz]}>
            {/* Solid volume */}
            <mesh>
              <boxGeometry args={[sx, sz, sy]} />
              <meshStandardMaterial
                color={style.base}
                emissive={style.emissive}
                emissiveIntensity={0.3}
                transparent
                opacity={0.55}
                metalness={0.3}
                roughness={0.6}
              />
            </mesh>
            {/* Edge wireframe glow */}
            <lineSegments geometry={edgeGeo}>
              <lineBasicMaterial color={style.edge} transparent opacity={0.6} />
            </lineSegments>
          </group>
        );
      })}

      {/* Search zone indicators with dynamic sector completion feedback */}
      {searchZones.map((zone) => {
        const secInfo = sectors[zone.id];
        const pct = secInfo?.coverage_pct || 0;
        // Dynamic completion color:
        // >=80% = neon emerald (#10b981), >=40% = cyan (#00f0ff), otherwise standard zone color
        let zColor = zone.priority === 'high' ? '#f59e0b' : '#8b5cf6';
        if (pct >= 80) {
          zColor = '#10b981';
        } else if (pct >= 40) {
          zColor = '#00f0ff';
        }

        return (
          <group key={zone.id}>
            {/* Flat perimeter ring on ground */}
            <mesh
              position={[zone.center[0], 0.05, zone.center[1]]}
              rotation={[-Math.PI / 2, 0, 0]}
            >
              <ringGeometry args={[zone.radius - 0.5, zone.radius, 64]} />
              <meshBasicMaterial
                color={zColor}
                transparent
                opacity={0.35}
                side={THREE.DoubleSide}
              />
            </mesh>
            {/* Translucent volume cylinder */}
            <mesh position={[zone.center[0], wz / 4, zone.center[1]]}>
              <cylinderGeometry args={[zone.radius, zone.radius, wz / 2, 32, 1, true]} />
              <meshBasicMaterial
                color={zColor}
                transparent
                opacity={0.06}
                side={THREE.DoubleSide}
                wireframe
              />
            </mesh>
          </group>
        );
      })}

      {/* Basecamp spawn marker */}
      <group position={[10, 0.1, 10]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[4.5, 5, 32]} />
          <meshBasicMaterial color="#10b981" transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, 2.5, 0]}>
          <cylinderGeometry args={[0.15, 0.15, 5, 8]} />
          <meshBasicMaterial color="#10b981" transparent opacity={0.3} />
        </mesh>
        <mesh position={[0, 5.2, 0]}>
          <sphereGeometry args={[0.3, 8, 8]} />
          <meshBasicMaterial color="#10b981" />
        </mesh>
      </group>
    </group>
  );
}

export default WorldEnvironment;
