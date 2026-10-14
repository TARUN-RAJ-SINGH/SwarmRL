/**
 * SwarmRL — Day 13: Dynamic 2D Ground Coverage Heatmap
 *
 * Renders a dynamic glowing thermal/radar coverage heatmap on the terrain
 * from the 25x25 density grid streamed from the backend.
 * Uses an offscreen HTML Canvas updated per frame and uploaded to a Three.js CanvasTexture.
 */

import React, { useMemo, useEffect, useRef } from 'react';
import * as THREE from 'three';

export function CoverageHeatmap({ heatmapGrid, worldConfig, visible = true }) {
  const textureRef = useRef(null);
  const canvasRef = useRef(null);

  const world = worldConfig?.world || { x_min: 0, x_max: 100, y_min: 0, y_max: 100 };
  const wx = world.x_max - world.x_min;
  const wy = world.y_max - world.y_min;
  const cx = world.x_min + wx / 2;
  const cz = world.y_min + wy / 2;

  // Initialize offscreen 2D canvas for texture generation
  const { canvas, ctx } = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 100;
    c.height = 100;
    const context = c.getContext('2d');
    canvasRef.current = c;
    return { canvas: c, ctx: context };
  }, []);

  // Initialize Three.js texture
  const texture = useMemo(() => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    textureRef.current = tex;
    return tex;
  }, [canvas]);

  // Update canvas texture whenever heatmapGrid updates
  useEffect(() => {
    if (!ctx || !heatmapGrid || !heatmapGrid.length) return;

    const rows = heatmapGrid.length;
    const cols = heatmapGrid[0].length;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    // Clear background to fully transparent
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw heat cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = heatmapGrid[r][c];
        if (val > 0.01) {
          // Heat color interpolation:
          // Low: cyan/blue (0, 240, 255), High: neon emerald green (16, 185, 129)
          const alpha = Math.min(val * 0.75 + 0.15, 0.85);
          if (val > 0.6) {
            ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;
          } else {
            ctx.fillStyle = `rgba(0, 240, 255, ${alpha})`;
          }
          // Note: Three.js plane orientation - match simulation coords
          ctx.fillRect(c * cellW, (rows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    if (textureRef.current) {
      textureRef.current.needsUpdate = true;
    }
  }, [heatmapGrid, ctx, canvas]);

  if (!visible) return null;

  return (
    <mesh
      position={[cx, 0.04, cz]}
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <planeGeometry args={[wx, wy]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.8}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

export default CoverageHeatmap;
