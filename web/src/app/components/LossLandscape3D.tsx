'use client';

import { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { Html, OrbitControls, Grid, PerspectiveCamera, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useI18n } from '../i18n';
import { useTheme } from '../theme';
import { getViridisColorNormalized } from '../lib/colormap';

interface LossLandscape3DProps {
  X: number[][];
  Y: number[][];
  lossGrid: number[][];
  baselineLoss?: number;
  trajectory?: {
    traj_1: number[];
    traj_2: number[];
    traj_3?: number[];
    epochs: number[];
    losses?: number[];
  };
  trajectoryHighlight?: {
    traj_1: number[];
    traj_2: number[];
    traj_3?: number[];
    epochs: number[];
  };
  /**
   * Optional axis labels, so Slice view can repurpose the component for
   * β–γ / α–γ planes while keeping Y as loss.
   * Defaults: XLabel='α', planeLabel='β'.
   */
  xLabel?: string;
  planeLabel?: string;
}

type Normalizer = {
  useLogZ: boolean;
  zEps: number;
  zScale: number;
  xCenter: number;
  yCenter: number;
  xyScale: number;
  zMinT: number;
  zMaxT: number;
  toX: (x: number) => number;
  toY: (y: number) => number;
  toZ: (z: number) => number;
  stats: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    zMin: number;
    zMax: number;
    zMinT: number;
    zMaxT: number;
  };
};

function buildNormalizer(X: number[][], Y: number[][], lossGrid: number[][], useLogZ: boolean): Normalizer {
  const flatX = X.flat();
  const flatY = Y.flat();
  const flatZ = lossGrid.flat().filter((v) => Number.isFinite(v));

  const xMin = Math.min(...flatX);
  const xMax = Math.max(...flatX);
  const yMin = Math.min(...flatY);
  const yMax = Math.max(...flatY);
  const zMin = Math.min(...flatZ);
  const zMax = Math.max(...flatZ);

  const xCenter = (xMin + xMax) / 2;
  const yCenter = (yMin + yMax) / 2;
  const xHalf = Math.max(Math.abs(xMax - xCenter), 1e-9);
  const yHalf = Math.max(Math.abs(yMax - yCenter), 1e-9);
  const xyScale = Math.max(xHalf, yHalf);

  const zEps = 1e-12;
  const zScale = 0.85; // vertical exaggeration (0..1)

  const zTransform = (z: number) => (useLogZ ? Math.log10(Math.max(z, 0) + zEps) : z);
  const zMinT = zTransform(zMin);
  const zMaxT = zTransform(zMax);
  const zRangeT = zMaxT - zMinT || 1;

  const toX = (x: number) => (x - xCenter) / xyScale;
  const toY = (y: number) => (y - yCenter) / xyScale;
  const toZ = (z: number) => ((zTransform(z) - zMinT) / zRangeT) * zScale;

  return {
    useLogZ,
    zEps,
    zScale,
    xCenter,
    yCenter,
    xyScale,
    zMinT,
    zMaxT,
    toX,
    toY,
    toZ,
    stats: { xMin, xMax, yMin, yMax, zMin, zMax, zMinT, zMaxT },
  };
}

function LossSurface({
  X,
  Y,
  lossGrid,
  norm,
  onHover,
}: {
  X: number[][];
  Y: number[][];
  lossGrid: number[][];
  norm: Normalizer;
  onHover?: (info: { alpha: number; beta: number; loss: number } | null) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Create geometry from grid data (normalized into a stable [-1,1] view space)
  const geometry = useMemo(() => {
    // Defensive: ensure grids are well-formed
    if (!Array.isArray(X) || !Array.isArray(Y) || !Array.isArray(lossGrid)) {
      return new THREE.BufferGeometry();
    }
    const rows = Math.min(X.length, Y.length, lossGrid.length);
    const cols = Math.min(X[0]?.length || 0, Y[0]?.length || 0, lossGrid[0]?.length || 0);
    if (rows === 0 || cols === 0) {
      return new THREE.BufferGeometry();
    }

    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const indices: number[] = [];
    const colors: number[] = [];

    // Normalize loss values for color mapping
    const flatZ = lossGrid.flat().filter((v) => Number.isFinite(v));
    const minZ = Math.min(...flatZ);
    const maxZ = Math.max(...flatZ);
    const rangeZ = maxZ - minZ || 1;

    // Create vertices
    // Three.js coordinates: X=right, Y=up, Z=forward
    // We want: α→X (right), loss→Y (up), β→Z (forward/depth)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // Guard against ragged rows
        const lx = X[i]?.[j];
        const ly = Y[i]?.[j];
        const lz = lossGrid[i]?.[j];
        if (!Number.isFinite(lx) || !Number.isFinite(ly) || !Number.isFinite(lz)) continue;

        const alpha = norm.toX(lx);
        const beta = norm.toY(ly);
        const loss = norm.toZ(lz);
        
        // Map: α→X, loss→Y (up!), β→Z
        vertices.push(alpha, loss, beta);

        // Color based on loss value (viridis colormap)
        const normalizedLoss = (lz - minZ) / rangeZ;
        const color = getViridisColorNormalized(normalizedLoss);
        colors.push(color.r, color.g, color.b);
      }
    }

    // Create indices for triangles
    for (let i = 0; i < rows - 1; i++) {
      for (let j = 0; j < cols - 1; j++) {
        const a = i * cols + j;
        const b = i * cols + (j + 1);
        const c = (i + 1) * cols + j;
        const d = (i + 1) * cols + (j + 1);

        // Two triangles per quad
        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    return geometry;
  }, [X, Y, lossGrid, norm]);

  const handlePointerMove = (event: any) => {
    if (!onHover || !event.point) return;
    
    // event.point is the intersection point in 3D space (already normalized)
    // We need to convert it back to original data space
    const normalizedX = event.point.x; // α in normalized space
    const normalizedZ = event.point.z; // β in normalized space  
    const normalizedY = event.point.y; // loss in normalized space
    
    // Convert back to original data space
    const alpha = normalizedX * norm.xyScale + norm.xCenter;
    const beta = normalizedZ * norm.xyScale + norm.yCenter;
    
    // Find the closest grid point to interpolate loss
    let closestLoss = 0;
    let minDist = Infinity;
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[0].length; j++) {
        const dx = X[i][j] - alpha;
        const dy = Y[i][j] - beta;
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          closestLoss = lossGrid[i][j];
        }
      }
    }
    
    onHover({ alpha, beta, loss: closestLoss });
  };

  const handlePointerOut = () => {
    if (onHover) {
      onHover(null);
    }
  };

  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  );
}

function TrajectoryLine({
  trajectory,
  X,
  Y,
  lossGrid,
  norm,
}: { 
  trajectory: { traj_1: number[]; traj_2: number[]; traj_3?: number[]; epochs: number[]; losses?: number[] };
  X: number[][];
  Y: number[][];
  lossGrid: number[][];
  norm: Normalizer;
}) {
  const { points, epochs, trainingLosses, landscapeLosses } = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const epochs: number[] = [];
    const trainingLosses: (number | null)[] = [];
    const landscapeLosses: number[] = [];
    const { traj_1, traj_2, traj_3, losses: trajectoryLosses } = trajectory;

    console.log('[TrajectoryLine] Building trajectory points:', {
      traj1Length: traj_1?.length,
      traj2Length: traj_2?.length,
      traj3Length: traj_3?.length,
      hasTraj3: !!traj_3,
    });

    // Interpolate loss values along trajectory
    // traj_1, traj_2 are projections onto dir1, dir2 (α, β coordinates)
    // traj_3 (if exists) is projection onto dir3 (γ coordinate for 3D mode)
    // For 3D Surface view, we use 2D loss grid, so we ignore traj_3 and interpolate from 2D grid
    // Map coordinates: α→X, loss→Y (up!), β→Z
    
    // Get grid bounds for validation
    const flatX = X.flat();
    const flatY = Y.flat();
    const xMin = Math.min(...flatX);
    const xMax = Math.max(...flatX);
    const yMin = Math.min(...flatY);
    const yMax = Math.max(...flatY);
    
    // Build array of points with their epochs and loss values for sorting and debugging
    const pointData: Array<{ point: THREE.Vector3; epoch: number; index: number; loss: number; trainingLoss: number | null; alpha: number; beta: number }> = [];
    
    for (let i = 0; i < traj_1.length; i++) {
      const alpha = traj_1[i];
      const beta = traj_2[i];
      
      // Clamp coordinates to grid bounds to ensure visibility
      const alphaClamped = Math.max(xMin, Math.min(xMax, alpha));
      const betaClamped = Math.max(yMin, Math.min(yMax, beta));
      
      // Always interpolate loss from 2D grid (3D Surface shows 2D surface, not 3D volume)
      const landscapeLoss = interpolateLoss(alphaClamped, betaClamped, X, Y, lossGrid);
      
      // Get training loss if available
      const trainingLoss = trajectoryLosses && i < trajectoryLosses.length ? trajectoryLosses[i] : null;
      
      const x = norm.toX(alphaClamped);
      const y = norm.toZ(landscapeLoss) + 0.002; // Tiny offset to prevent Z-fighting
      const z = norm.toY(betaClamped);
      
      // Validate that point is within reasonable bounds
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        const epoch = trajectory.epochs[i] ?? i;
        pointData.push({
          point: new THREE.Vector3(x, y, z),
          epoch,
          index: i,
          loss: landscapeLoss,
          trainingLoss,
          alpha: alphaClamped,
          beta: betaClamped,
        });
      }
    }

    // Sort by epoch (low to high) to ensure correct trajectory order
    pointData.sort((a, b) => a.epoch - b.epoch);

    // Extract sorted points, epochs, and losses
    for (const { point, epoch, loss, trainingLoss } of pointData) {
      points.push(point);
      epochs.push(epoch);
      landscapeLosses.push(loss);
      trainingLosses.push(trainingLoss);
    }

    // Debug: Check first and last epoch loss values
    const firstPointData = pointData[0];
    const lastPointData = pointData[pointData.length - 1];
    
    console.log('[TrajectoryLine] Generated points:', {
      pointCount: points.length,
      firstPoint: points[0],
      lastPoint: points[points.length - 1],
      firstEpoch: epochs[0],
      lastEpoch: epochs[epochs.length - 1],
      firstLoss: firstPointData?.loss,
      lastLoss: lastPointData?.loss,
      firstAlpha: firstPointData?.alpha,
      firstBeta: firstPointData?.beta,
      lastAlpha: lastPointData?.alpha,
      lastBeta: lastPointData?.beta,
      lossComparison: firstPointData && lastPointData ? {
        firstLoss: firstPointData.loss,
        lastLoss: lastPointData.loss,
        lossDecreased: firstPointData.loss > lastPointData.loss,
        lossIncreased: firstPointData.loss < lastPointData.loss,
      } : null,
      normStats: {
        xRange: [norm.stats.xMin, norm.stats.xMax],
        yRange: [norm.stats.yMin, norm.stats.yMax],
        zRange: [norm.stats.zMin, norm.stats.zMax],
      },
    });

    return { points, epochs, trainingLosses, landscapeLosses };
  }, [trajectory, X, Y, lossGrid, norm]);

  // Sample points to show (every N-th point to avoid clutter)
  const sampledIndices = useMemo(() => {
    const total = points.length;
    if (total <= 20) return Array.from({ length: total }, (_, i) => i);
    
    // Show first, last, and evenly spaced points
    const step = Math.ceil(total / 20);
    const indices: number[] = [0]; // Always show first
    for (let i = step; i < total - 1; i += step) {
      indices.push(i);
    }
    indices.push(total - 1); // Always show last
    return indices;
  }, [points]);

  // Don't render if we don't have enough points (Line needs at least 2 points)
  if (points.length < 2) {
    return null;
  }

  return (
    <>
      {/* Trajectory line */}
      <Line points={points} color="#ff4444" lineWidth={3} />
      
      {/* All trajectory points as small spheres */}
      {sampledIndices.map((i) => {
        const progress = i / (points.length - 1);
        // Color gradient: green -> yellow -> red
        const color = new THREE.Color(
          progress < 0.5 
            ? THREE.MathUtils.lerp(0.2, 1.0, progress * 2)  // green to yellow
            : 1.0,
          progress < 0.5 
            ? THREE.MathUtils.lerp(0.8, 1.0, progress * 2)  // green to yellow
            : THREE.MathUtils.lerp(1.0, 0.2, (progress - 0.5) * 2), // yellow to red
          0.2
        );
        
        const isStart = i === 0;
        const isEnd = i === points.length - 1;
        const radius = isStart || isEnd ? 0.04 : 0.02;
        const epoch = epochs[i];
        const trainingLoss = trainingLosses[i];
        const landscapeLoss = landscapeLosses[i];
        
        return (
          <mesh key={i} position={points[i]}>
            <octahedronGeometry args={[radius * 1.2, 0]} />
            <meshStandardMaterial 
              color={color} 
              emissive={color}
              emissiveIntensity={isStart || isEnd ? 0.3 : 0.1}
            />
          </mesh>
        );
      })}
      
      {/* Start point marker (larger) */}
      {points[0] && (
        <mesh position={points[0]}>
          <octahedronGeometry args={[0.07, 0]} />
          <meshStandardMaterial 
            color="#00ff00" 
            emissive="#00ff00"
            emissiveIntensity={0.5}
          />
        </mesh>
      )}
      
      {/* End point marker (larger) */}
      {points[points.length - 1] && (
        <mesh position={points[points.length - 1]}>
          <octahedronGeometry args={[0.07, 0]} />
          <meshStandardMaterial 
            color="#ff0000" 
            emissive="#ff0000"
            emissiveIntensity={0.5}
          />
        </mesh>
      )}
    </>
  );
}

function interpolateLoss(x: number, y: number, X: number[][], Y: number[][], lossGrid: number[][]): number {
  // Improved bilinear interpolation with proper grid handling
  const rows = X.length;
  const cols = X[0]?.length || 0;
  
  if (rows === 0 || cols === 0) {
    return 0;
  }

  // Validate lossGrid dimensions match X and Y
  // Allow flexible dimensions - sometimes lossGrid might be [N][M] while X/Y are [N+1][M+1] or vice versa
  // Just check if we have enough rows to do interpolation
  if (!lossGrid || lossGrid.length < 2) {
    return 0;
  }
  
  // Use the minimum common dimensions
  const safeRows = Math.min(rows, lossGrid.length);
  const safeCols = Math.min(cols, lossGrid[0]?.length || 0);
  
  if (safeRows < 2 || safeCols < 2) {
      return 0;
  }

  // Find the closest grid cell using distance
  let minDist = Infinity;
  let bestValue = lossGrid[0]?.[0] ?? 0;
  
  // First, try to find exact cell
  for (let i = 0; i < safeRows; i++) {
    // Check if lossGrid[i] exists and has the right length
    if (!lossGrid[i]) {
      continue; // Skip this row if it's invalid
    }
    const rowLen = Math.min(cols, lossGrid[i].length);
    
    for (let j = 0; j < rowLen; j++) {
      // Validate X and Y arrays
      if (!X[i] || !Y[i] || X[i][j] === undefined || Y[i][j] === undefined) {
        continue; // Skip invalid cells
      }
      
      const gridX = X[i][j];
      const gridY = Y[i][j];
      const dist = Math.sqrt((x - gridX) ** 2 + (y - gridY) ** 2);
      
      if (dist < minDist) {
        minDist = dist;
        const lossValue = lossGrid[i][j];
        if (lossValue !== undefined && Number.isFinite(lossValue)) {
          bestValue = lossValue;
        }
      }
    }
  }

  // Try bilinear interpolation if we're within a reasonable cell
  for (let i = 0; i < safeRows - 1; i++) {
    // Validate row exists in all arrays
    if (!lossGrid[i] || !lossGrid[i + 1] || !X[i] || !X[i + 1] || !Y[i] || !Y[i + 1]) {
      continue;
    }
    
    const rowLen = Math.min(safeCols, lossGrid[i].length, lossGrid[i+1].length);
    
    for (let j = 0; j < rowLen - 1; j++) {
      // Validate all required cells exist
      if (
        lossGrid[i][j] === undefined || lossGrid[i][j + 1] === undefined ||
        lossGrid[i + 1][j] === undefined || lossGrid[i + 1][j + 1] === undefined ||
        X[i][j] === undefined || X[i][j + 1] === undefined ||
        X[i + 1][j] === undefined || X[i + 1][j + 1] === undefined ||
        Y[i][j] === undefined || Y[i][j + 1] === undefined ||
        Y[i + 1][j] === undefined || Y[i + 1][j + 1] === undefined
      ) {
        continue; // Skip incomplete cells
      }
      
      const x00 = X[i][j];
      const y00 = Y[i][j];
      const x01 = X[i][j + 1];
      const y01 = Y[i][j + 1];
      const x10 = X[i + 1][j];
      const y10 = Y[i + 1][j];
      const x11 = X[i + 1][j + 1];
      const y11 = Y[i + 1][j + 1];

      // Check if point is within this cell (with some tolerance)
      const xMin = Math.min(x00, x01, x10, x11);
      const xMax = Math.max(x00, x01, x10, x11);
      const yMin = Math.min(y00, y01, y10, y11);
      const yMax = Math.max(y00, y01, y10, y11);

      if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
        // Bilinear interpolation
        const z00 = lossGrid[i][j];
        const z01 = lossGrid[i][j + 1];
        const z10 = lossGrid[i + 1][j];
        const z11 = lossGrid[i + 1][j + 1];
        
        // Validate all loss values are finite
        if (!Number.isFinite(z00) || !Number.isFinite(z01) || 
            !Number.isFinite(z10) || !Number.isFinite(z11)) {
          continue; // Skip if any value is invalid
        }

        // Normalize coordinates within cell
        const dx = xMax > xMin ? (x - xMin) / (xMax - xMin) : 0.5;
        const dy = yMax > yMin ? (y - yMin) / (yMax - yMin) : 0.5;

        // Use triangle-based interpolation to match the 3D mesh geometry.
        // The mesh splits the quad into two triangles:
        // T1: (0,0)-(0,1)-(1,0)  => matches dx + dy <= 1 (if dx,dy are 0..1 based on corners)
        // T2: (0,1)-(1,1)-(1,0)  => matches dx + dy > 1
        // Note: Indices in geometry are a(0,0), b(0,1), c(1,0), d(1,1).
        // Triangles are (a,b,c) and (b,d,c).
        // Triangle 1 (a,b,c): (0,0), (0,1), (1,0). Plane z = z00 + (z10-z00)*dx + (z01-z00)*dy
        // Triangle 2 (b,d,c): (0,1), (1,1), (1,0). Plane z = z11 + (z01-z11)*(1-dx) + (z10-z11)*(1-dy)

        let z = 0;
        
        // Check if we are in the "upper-left" triangle (in standard image coords) or "lower-right"
        // Note: dx is along i (rows), dy is along j (cols) in our loops? 
        // Wait, X[i][j] vs X[i+1][j]. i is usually row. 
        // In geometry generation:
        // a = i,j; b = i,j+1; c = i+1,j; d = i+1,j+1.
        // dx moves from i to i+1. dy moves from j to j+1.
        // T1: (i,j), (i,j+1), (i+1,j) => (0,0), (0,1), (1,0) in (dx, dy) space.
        // Condition for T1: dx + dy <= 1?
        // Line (0,1) to (1,0) is x + y = 1.
        // Origin (0,0) is satisfying 0+0 <= 1. So yes.
        
        if (dx + dy <= 1) {
          // Triangle 1: (0,0), (0,1), (1,0) -> z00, z01, z10
          // z = z00 + (z10 - z00) * dx + (z01 - z00) * dy
          z = z00 + (z10 - z00) * dx + (z01 - z00) * dy;
        } else {
          // Triangle 2: (0,1), (1,1), (1,0) -> z01, z11, z10
          // We interpolate from (1,1) back.
          // z = z11 + (z01 - z11) * (1 - dx) + (z10 - z11) * (1 - dy)
          z = z11 + (z01 - z11) * (1 - dx) + (z10 - z11) * (1 - dy);
        }

        return z;
      }
    }
  }

  // Fallback: return closest value
  return bestValue;
}

function fmtLoss(x: number) {
  return Number.isFinite(x) ? x.toExponential(3) : String(x);
}

function TrajectoryHighlight({
  trajectoryHighlight,
  X,
  Y,
  lossGrid,
  norm,
  viewEpoch,
}: {
  trajectoryHighlight: { traj_1: number[]; traj_2: number[]; epochs: number[] };
  X: number[][];
  Y: number[][];
  lossGrid: number[][];
  norm: Normalizer;
  viewEpoch: number;
}) {
  const points = useMemo(() => {
    if (!trajectoryHighlight || !trajectoryHighlight.traj_1) return [];
    
    const pts: THREE.Vector3[] = [];
    const { traj_1, traj_2, epochs } = trajectoryHighlight;
    const len = Math.min(traj_1.length, traj_2.length);
    
    // Bounds for clamping
    const flatX = X.flat();
    const flatY = Y.flat();
    const xMin = Math.min(...flatX);
    const xMax = Math.max(...flatX);
    const yMin = Math.min(...flatY);
    const yMax = Math.max(...flatY);

    for (let i = 0; i < len; i++) {
      const ep = epochs[i] ?? i;
      if (ep > viewEpoch) continue;

      const alpha = traj_1[i];
      const beta = traj_2[i];
      
      const alphaClamped = Math.max(xMin, Math.min(xMax, alpha));
      const betaClamped = Math.max(yMin, Math.min(yMax, beta));
      
      const loss = interpolateLoss(alphaClamped, betaClamped, X, Y, lossGrid);
      
      const x = norm.toX(alphaClamped);
      const y = norm.toZ(loss) + 0.015; // Higher than line to avoid clipping
      const z = norm.toY(betaClamped);
      
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        pts.push(new THREE.Vector3(x, y, z));
      }
    }
    return pts;
  }, [trajectoryHighlight, X, Y, lossGrid, norm, viewEpoch]);

  if (points.length === 0) return null;

  return (
    <group>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          {/* Octahedron looks like a 3D Diamond, aligning with the 2D view style */}
          <octahedronGeometry args={[0.03, 0]} />
          <meshStandardMaterial 
            color="#00ffff" 
            emissive="#00ffff" 
            emissiveIntensity={0.8}
            roughness={0.1}
            metalness={0.9} 
          />
        </mesh>
      ))}
    </group>
  );
}

export default function LossLandscape3D({
  X,
  Y,
  lossGrid,
  baselineLoss,
  trajectory,
  trajectoryHighlight,
  xLabel,
  planeLabel,
}: LossLandscape3DProps) {
  const { t } = useI18n();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const ui = useMemo(() => {
    return isDark
      ? {
          panelBg: 'rgba(0,0,0,0.75)',
          panelBorder: 'rgba(255,255,255,0.30)',
          panelShadow: '0 12px 40px rgba(0,0,0,0.5)',
          text: 'white',
          surface: 'rgba(255,255,255,0.05)',
          surfaceBorder: 'rgba(255,255,255,0.10)',
          divider: 'rgba(255,255,255,0.20)',
          tooltipBg: 'rgba(0,0,0,0.85)',
          tooltipText: 'white',
        }
      : {
          panelBg: 'rgba(255,255,255,0.86)',
          panelBorder: 'rgba(15,23,42,0.12)',
          panelShadow: '0 12px 40px rgba(15,23,42,0.12)',
          text: '#0f172a',
          surface: 'rgba(15,23,42,0.04)',
          surfaceBorder: 'rgba(15,23,42,0.10)',
          divider: 'rgba(15,23,42,0.12)',
          tooltipBg: 'rgba(255,255,255,0.92)',
          tooltipText: '#0f172a',
        };
  }, [isDark]);
  const [useLog, setUseLog] = useState(true);
  const [hoverInfo, setHoverInfo] = useState<{ alpha: number; beta: number; loss: number } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null); // canvas-local coords
  const [legendPos, setLegendPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingLegend = useRef(false);
  const dragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const posStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const norm = useMemo(() => buildNormalizer(X, Y, lossGrid, useLog), [X, Y, lossGrid, useLog]);
  const zLabel = norm.useLogZ ? t.log10Loss : t.loss;
  const xLabelText = xLabel ?? 'α';
  const planeLabelText = planeLabel ?? 'β';

  // Compute epoch range
  const { minEpoch, maxEpoch } = useMemo(() => {
    if (!trajectory || !trajectory.epochs || trajectory.epochs.length === 0) {
      return { minEpoch: 0, maxEpoch: 0 };
    }
    return {
      minEpoch: Math.min(...trajectory.epochs),
      maxEpoch: Math.max(...trajectory.epochs),
    };
  }, [trajectory]);

  const [viewEpoch, setViewEpoch] = useState<number>(maxEpoch);

  // Update viewEpoch if maxEpoch changes
  useMemo(() => {
    setViewEpoch(maxEpoch);
  }, [maxEpoch]);

  // Legend drag handlers to avoid blocking top-right content - optimized with requestAnimationFrame
  useEffect(() => {
    let rafId: number | null = null;
    
    const onMove = (e: MouseEvent) => {
      if (!isDraggingLegend.current) return;
      
      // Throttle updates with requestAnimationFrame
      if (rafId !== null) return;
      
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setLegendPos({ x: posStart.current.x + dx, y: posStart.current.y + dy });
      });
    };
    
    const onUp = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (isDraggingLegend.current) isDraggingLegend.current = false;
    };
    
    window.addEventListener('mousemove', onMove, { passive: true });
    window.addEventListener('mouseup', onUp);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  const startLegendDrag = (e: React.MouseEvent) => {
    isDraggingLegend.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    posStart.current = legendPos;
    e.preventDefault();
  };

  // Filter trajectory based on viewEpoch
  const filteredTrajectory = useMemo(() => {
    if (!trajectory) return undefined;
    const len = trajectory.traj_1.length;
    const filtered = {
      traj_1: [] as number[],
      traj_2: [] as number[],
      traj_3: trajectory.traj_3 ? [] as number[] : undefined,
      epochs: [] as number[],
    };
    for (let i = 0; i < len; i++) {
      const ep = trajectory.epochs[i] ?? i;
      if (ep > viewEpoch) continue;
      filtered.traj_1.push(trajectory.traj_1[i]);
      filtered.traj_2.push(trajectory.traj_2[i]);
      if (filtered.traj_3 && trajectory.traj_3) filtered.traj_3.push(trajectory.traj_3[i]);
      filtered.epochs.push(ep);
    }
    return filtered;
  }, [trajectory, viewEpoch]);

  // Debug: log trajectory data
  useEffect(() => {
    console.log('[LossLandscape3D] Trajectory data:', {
      hasTrajectory: !!trajectory,
      hasTraj1: !!trajectory?.traj_1,
      traj1Length: trajectory?.traj_1?.length,
      traj2Length: trajectory?.traj_2?.length,
      traj3Length: trajectory?.traj_3?.length,
      epochsLength: trajectory?.epochs?.length,
      trajectory,
    });
  }, [trajectory]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Colorbar Legend (identical to 2D view) */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          right: 16,
          transform: `translate(${legendPos.x}px, ${legendPos.y}px)`,
          padding: '18px 20px',
          borderRadius: 18,
          border: `1px solid ${ui.panelBorder}`,
          background: ui.panelBg,
          backdropFilter: 'blur(12px)',
          color: ui.text,
          zIndex: 10,
          fontSize: 13,
          lineHeight: 1.6,
          width: 300,
          boxShadow: ui.panelShadow,
          cursor: isDraggingLegend.current ? 'grabbing' : 'grab',
        }}
        onMouseDown={startLegendDrag}
      >
        <div style={{ 
          fontWeight: 800, 
          marginBottom: 16, 
          fontSize: 15, 
          letterSpacing: '0.02em',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span>🌐</span>
          <span>{t.lossLandscape3D}</span>
        </div>
        
        <div style={{ 
          display: 'flex', 
          gap: 14, 
          marginBottom: 16,
          alignItems: 'flex-start',
        }}>
          {/* Colorbar */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div
              title={t.lowerIsBetter}
              style={{
                width: 24,
                height: 160,
                borderRadius: 0,
                border: 'none',
                background: `linear-gradient(to top, ${(() => {
                  const steps = 20; // More steps for smoother gradient
                  const colors: string[] = [];
                  for (let i = 0; i <= steps; i++) {
                    // Ensure exact boundaries: t=0 for first, t=1 for last
                    const t = i === 0 ? 0 : i === steps ? 1 : i / steps;
                    const c = getViridisColorNormalized(t);
                    const rgb = `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
                    const stopPercent = i === 0 ? '0%' : i === steps ? '100%' : `${(i / steps * 100).toFixed(2)}%`;
                    colors.push(`${rgb} ${stopPercent}`);
                  }
                  return colors.join(', ');
                })()})`,
                boxShadow: isDark
                  ? '0 4px 12px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,255,255,0.1)'
                  : '0 4px 12px rgba(15,23,42,0.10), inset 0 0 20px rgba(255,255,255,0.25)',
                cursor: 'help',
                position: 'relative',
              }}
            />
            <div style={{ 
              fontSize: 9, 
              opacity: isDark ? 0.6 : 0.75, 
              marginTop: 4,
              textAlign: 'center',
              lineHeight: 1.2,
            }}>
              {t.hoverForHint}
            </div>
          </div>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 160 }}>
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(255,100,100,0.15)' : 'rgba(239,68,68,0.10)',
              border: isDark ? '1px solid rgba(255,100,100,0.3)' : '1px solid rgba(239,68,68,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.highLoss}</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#ff6b6b' }}>{fmtLoss(norm.stats.zMax)}</div>
            </div>
            
            <div style={{ 
              textAlign: 'center',
              padding: '6px 0',
              fontSize: 10,
              opacity: isDark ? 0.7 : 0.8,
              fontStyle: 'italic',
            }}>
              {t.lowerIsBetterArrow}
            </div>
            
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(100,255,150,0.15)' : 'rgba(16,185,129,0.10)',
              border: isDark ? '1px solid rgba(100,255,150,0.3)' : '1px solid rgba(16,185,129,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.lowLoss}</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#51cf66' }}>{fmtLoss(norm.stats.zMin)}</div>
            </div>
          </div>
        </div>

        <div style={{ 
          borderTop: `2px solid ${ui.divider}`, 
          paddingTop: 14, 
          marginTop: 14, 
          display: 'grid', 
          gap: 10 
        }}>
          {baselineLoss !== undefined && (
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: ui.surface,
            }}>
              <span style={{ opacity: 0.9, fontWeight: 600 }}>📍 {t.baselineLoss}</span>
              <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#fbbf24' }}>{fmtLoss(baselineLoss)}</span>
            </div>
          )}
          
          {trajectory && (
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 10, 
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(255, 80, 80, 0.1)' : 'rgba(239, 68, 68, 0.08)',
              border: isDark ? '1px solid rgba(255, 80, 80, 0.3)' : '1px solid rgba(239, 68, 68, 0.18)',
            }}>
              <div style={{ 
                width: 24, 
                height: 4, 
                background: 'rgba(255, 80, 80, 1)', 
                borderRadius: 2,
                boxShadow: '0 0 8px rgba(255, 80, 80, 0.5)',
              }} />
              <span style={{ opacity: 0.95, fontWeight: 600 }}>{t.trainingTrajectory}</span>
            </div>
          )}
          
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            fontSize: 12,
            padding: '8px 10px',
            borderRadius: 8,
            background: ui.surface,
          }}>
            <span style={{ opacity: 0.9, fontWeight: 600 }}>{t.scale}</span>
            <button
              type="button"
              onClick={() => setUseLog((v) => !v)}
              style={{
                border: `2px solid ${
                  useLog
                    ? 'rgba(251, 191, 36, 0.55)'
                    : isDark
                      ? 'rgba(255,255,255,0.30)'
                      : 'rgba(15,23,42,0.18)'
                }`,
                background: useLog
                  ? 'rgba(251, 191, 36, 0.20)'
                  : isDark
                    ? 'rgba(255,255,255,0.08)'
                    : 'rgba(15,23,42,0.05)',
                color: ui.text,
                borderRadius: 8,
                padding: '6px 14px',
                cursor: 'pointer',
                fontWeight: 700,
                fontSize: 11,
                transition: 'all 0.2s ease',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = useLog
                  ? 'rgba(251, 191, 36, 0.28)'
                  : isDark
                    ? 'rgba(255,255,255,0.12)'
                    : 'rgba(15,23,42,0.07)')
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = useLog
                  ? 'rgba(251, 191, 36, 0.20)'
                  : isDark
                    ? 'rgba(255,255,255,0.08)'
                    : 'rgba(15,23,42,0.05)')
              }
            >
              {useLog ? `📊 ${t.log}` : `📈 ${t.linear}`}
            </button>
          </div>

          {trajectory && maxEpoch > 0 && (
            <div 
              style={{ 
                display: 'flex', 
                gap: 12, 
                alignItems: 'center', 
                fontSize: 12,
                padding: '8px 10px',
                borderRadius: 8,
                background: 'rgba(255,255,255,0.05)',
                marginTop: 8,
              }}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <span style={{ opacity: 0.9, fontWeight: 600 }}>{t.epoch}:</span>
              <input
                type="range"
                min={minEpoch}
                max={maxEpoch}
                step={1}
                value={viewEpoch}
                onChange={(e) => setViewEpoch(parseInt(e.target.value, 10))}
                style={{ 
                  flex: 1,
                  accentColor: '#ff4444',
                }}
              />
              <span style={{ 
                minWidth: 50, 
                textAlign: 'right', 
                opacity: 0.9, 
                fontFamily: 'monospace', 
                fontSize: 12 
              }}>
                {viewEpoch}
              </span>
            </div>
          )}
        </div>
        
        <div style={{
          marginTop: 12,
          paddingTop: 12,
          borderTop: `1px solid ${ui.surfaceBorder}`,
          fontSize: 10,
          opacity: isDark ? 0.6 : 0.75,
          lineHeight: 1.4,
        }}>
          💡 <strong>{t.tip3D}</strong>
        </div>
      </div>

      <Canvas
        onPointerMove={(e) => {
          const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
          setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        }}
        onPointerLeave={() => {
          setMousePos(null);
          setHoverInfo(null);
        }}
      >
        <color attach="background" args={[isDark ? '#0b1220' : '#f1f5f9']} />
        {/* Camera positioned to show Y-axis (loss) pointing up */}
        <PerspectiveCamera makeDefault position={[2.5, 2.5, 2.5]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 8]} intensity={1} />
        <LossSurface 
          X={X} 
          Y={Y} 
          lossGrid={lossGrid} 
          norm={norm}
          onHover={setHoverInfo}
        />
        {filteredTrajectory && (
          <TrajectoryLine trajectory={filteredTrajectory} X={X} Y={Y} lossGrid={lossGrid} norm={norm} />
        )}
        {trajectoryHighlight && (
          <TrajectoryHighlight 
            trajectoryHighlight={trajectoryHighlight} 
            X={X} 
            Y={Y} 
            lossGrid={lossGrid} 
            norm={norm} 
            viewEpoch={viewEpoch}
          />
        )}

        {/* Data-aware ground grid (normalized) - XZ plane (ground) */}
        <Grid
          args={[2.4, 2.4]}
          cellSize={0.2}
          cellThickness={0.6}
          sectionSize={1.2}
          sectionThickness={1.0}
          fadeDistance={6}
          infiniteGrid={false}
          position={[0, 0, 0]}
          rotation={[-Math.PI / 2, 0, 0]}
        />

        {/* Proper axes with labels - Y pointing up (loss) */}
        <Line points={[[0, 0, 0], [1.25, 0, 0]]} color="#93c5fd" lineWidth={1} />
        <Line points={[[0, 0, 0], [0, 1.25, 0]]} color="#fca5a5" lineWidth={1} />
        <Line points={[[0, 0, 0], [0, 0, 1.25]]} color="#a7f3d0" lineWidth={1} />
        <Html
          position={[1.32, 0, 0]}
          style={{
            color: isDark ? 'rgba(255,255,255,0.9)' : 'rgba(15,23,42,0.9)',
            fontSize: 13,
            fontWeight: 600,
            textShadow: isDark ? '0 1px 3px rgba(0,0,0,0.5)' : '0 1px 2px rgba(15,23,42,0.15)',
          }}
        >
          {xLabelText}
        </Html>
        <Html
          position={[0, 1.32, 0]}
          style={{
            color: isDark ? 'rgba(255,255,255,0.9)' : 'rgba(15,23,42,0.9)',
            fontSize: 13,
            fontWeight: 600,
            textShadow: isDark ? '0 1px 3px rgba(0,0,0,0.5)' : '0 1px 2px rgba(15,23,42,0.15)',
          }}
        >
          {zLabel} ↑
        </Html>
        <Html
          position={[0, 0, 1.32]}
          style={{
            color: isDark ? 'rgba(255,255,255,0.9)' : 'rgba(15,23,42,0.9)',
            fontSize: 13,
            fontWeight: 600,
            textShadow: isDark ? '0 1px 3px rgba(0,0,0,0.5)' : '0 1px 2px rgba(15,23,42,0.15)',
          }}
        >
          {planeLabelText}
        </Html>

        <OrbitControls target={[0, norm.zScale * 0.4, 0]} />
      </Canvas>

      {/* Hover Tooltip */}
      {hoverInfo && mousePos && (
        <div
          style={{
            position: 'absolute',
            left: mousePos.x + 8,
            top: mousePos.y + 8,
            padding: '14px 16px',
            borderRadius: 14,
            border: '2px solid rgba(251, 191, 36, 0.5)',
            background: ui.tooltipBg,
            backdropFilter: 'blur(12px)',
            color: ui.tooltipText,
            fontSize: 12,
            lineHeight: 1.7,
            minWidth: 200,
            pointerEvents: 'none',
            boxShadow: isDark
              ? '0 8px 24px rgba(0,0,0,0.5), 0 0 20px rgba(251, 191, 36, 0.2)'
              : '0 8px 24px rgba(15,23,42,0.12), 0 0 18px rgba(251, 191, 36, 0.18)',
            zIndex: 1000,
          }}
        >
          <div style={{ 
            fontWeight: 800, 
            marginBottom: 10, 
            fontSize: 11, 
            opacity: 0.9, 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em',
            color: '#fbbf24',
          }}>
            📍 {t.currentPosition}
          </div>
          <div style={{ display: 'grid', gap: 6, fontFamily: 'monospace', fontSize: 12 }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              borderBottom: `1px solid ${ui.surfaceBorder}`,
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{xLabelText}:</span> 
              <span style={{ fontWeight: 700, color: '#93c5fd' }}>{hoverInfo.alpha.toFixed(4)}</span>
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              borderBottom: `1px solid ${ui.surfaceBorder}`,
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{planeLabelText}:</span> 
              <span style={{ fontWeight: 700, color: '#a7f3d0' }}>{hoverInfo.beta.toFixed(4)}</span>
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              marginTop: 2,
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{t.loss}:</span> 
              <span style={{ 
                fontWeight: 700, 
                color: hoverInfo.loss < (norm.stats.zMin + norm.stats.zMax) / 2 ? '#51cf66' : '#ff6b6b',
              }}>
                {fmtLoss(hoverInfo.loss)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

