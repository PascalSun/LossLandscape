'use client';

import { useMemo, useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useI18n } from '../i18n';
import { useTheme } from '../theme';
import { getViridisColor } from '../lib/colormap';

function fmtLoss(x: number) {
  return Number.isFinite(x) ? x.toExponential(3) : String(x);
}

function finiteMinMax3D(grid: number[][][]) {
  let minV = Number.POSITIVE_INFINITY;
  let maxV = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < grid.length; i++) {
    const plane = grid[i];
    if (!plane) continue;
    for (let j = 0; j < plane.length; j++) {
      const row = plane[j];
      if (!row) continue;
      for (let k = 0; k < row.length; k++) {
        const v = row[k];
        if (!Number.isFinite(v)) continue;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
  }
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {
    // Fall back to a safe range so downstream rendering doesn't explode.
    return { minV: 0, maxV: 1 };
  }
  if (minV === maxV) return { minV, maxV: minV + 1e-12 };
  return { minV, maxV };
}

function robustMinMax3D(
  grid: number[][][],
  qLow: number = 0.02,
  qHigh: number = 0.98,
  maxSamples: number = 50_000
) {
  const nx = grid.length;
  const ny = grid[0]?.length || 0;
  const nz = grid[0]?.[0]?.length || 0;
  if (nx * ny * nz <= 0) return finiteMinMax3D(grid);

  const total = nx * ny * nz;
  // sqrt-ish stride gives decent coverage while keeping sample size bounded.
  const stride = Math.max(1, Math.floor(Math.sqrt(total / maxSamples)));

  const vals: number[] = [];
  for (let i = 0; i < nx; i += stride) {
    const plane = grid[i];
    if (!plane) continue;
    for (let j = 0; j < ny; j += stride) {
      const row = plane[j];
      if (!row) continue;
      for (let k = 0; k < nz; k += stride) {
        const v = row[k];
        if (!Number.isFinite(v)) continue;
        vals.push(v);
        if (vals.length >= maxSamples) break;
      }
      if (vals.length >= maxSamples) break;
    }
    if (vals.length >= maxSamples) break;
  }

  if (vals.length < 20) return finiteMinMax3D(grid);
  vals.sort((a, b) => a - b);

  const lo = Math.max(0, Math.min(vals.length - 1, Math.floor(qLow * (vals.length - 1))));
  const hi = Math.max(0, Math.min(vals.length - 1, Math.floor(qHigh * (vals.length - 1))));
  const minV = vals[lo];
  const maxV = vals[Math.max(hi, lo + 1)];
  return minV === maxV ? { minV, maxV: minV + 1e-12 } : { minV, maxV };
}

function minMaxFlat2D(grid: number[][]) {
  let minV = Number.POSITIVE_INFINITY;
  let maxV = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < grid.length; i++) {
    const row = grid[i];
    if (!row) continue;
    for (let j = 0; j < row.length; j++) {
      const v = row[j];
      if (!Number.isFinite(v)) continue;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return { minV: -1, maxV: 1 };
  return { minV, maxV };
}

// Normalize Z input from backend into a 1D gamma axis:
// - Preferred: 1D numeric array [gamma_k]
// - Also support 3D grid Z[i][j][k] -> take Z[0][0][k]
function normalizeZAxis(Z: any, lossGrid3d: number[][][], X: number[][], Y: number[][]): number[] {
  const nz = lossGrid3d[0]?.[0]?.length || 0;
  if (nz <= 0) return [];

  if (Array.isArray(Z) && Z.length > 0) {
    const first = Z[0];
    if (typeof first === 'number' || typeof first === 'string') {
      const arr = (Z as any[]).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      if (arr.length === nz) return arr;
    }
    if (Array.isArray(first)) {
      const zLine = (Z[0]?.[0] ?? []) as any[];
      const arr = zLine.map((v) => Number(v)).filter((v) => Number.isFinite(v));
      if (arr.length === nz) return arr;
    }
  }

  // Fallback: derive symmetric range from X/Y extent
  const { minV: xMin, maxV: xMax } = minMaxFlat2D(X);
  const { minV: yMin, maxV: yMax } = minMaxFlat2D(Y);
  const extent = Math.max(Math.abs(xMin), Math.abs(xMax), Math.abs(yMin), Math.abs(yMax), 1e-9);
  console.warn('[LossVolumeRender3D] Z missing or mismatched; deriving from X/Y grid extent', {
    nz,
    extent,
    ZShape: Array.isArray(Z) ? 'array' : typeof Z,
  });
  return Array.from({ length: nz }, (_, i) => -extent + (i / Math.max(1, nz - 1)) * (2 * extent));
}

function sliceToTexture(
  slice: number[][],
  minV: number,
  maxV: number,
  threshold: number = 0,
  opts?: { scale?: 'linear' | 'log'; alphaGamma?: number; alphaFloor?: number }
) {
  const h = slice.length;
  const w = slice[0]?.length || 0;
  const data = new Uint8Array(w * h * 4);
  const range = maxV - minV || 1;
  const thresholdNorm = (threshold - minV) / range;

  const scale = opts?.scale ?? 'log';
  const alphaGamma = opts?.alphaGamma ?? 0.5;
  const alphaFloor = opts?.alphaFloor ?? 0.02;
  const zEps = 1e-12;

  const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
  const normalize = (raw: number) => {
    const lin = clamp01((raw - minV) / range);
    if (scale === 'linear') return lin;
    // Match other views: log10(max(z,0)+eps)
    const safeRaw = Math.max(0, raw);
    const safeMin = Math.max(0, minV);
    const safeMax = Math.max(0, maxV);
    const lo = Math.log10(safeMin + zEps);
    const hi = Math.log10(safeMax + zEps);
    const denom = hi - lo || 1;
    const v = (Math.log10(safeRaw + zEps) - lo) / denom;
    return clamp01(v);
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = normalize(slice[y][x]);
      const idx = (y * w + x) * 4;
      
      // Use viridis colormap
      const color = getViridisColor(v);
      data[idx + 0] = color.r;
      data[idx + 1] = color.g;
      data[idx + 2] = color.b;
      
      // Alpha: higher loss = more opaque, with threshold filtering
      // Use a tunable curve + small floor so low values don't disappear.
      const t = v > thresholdNorm ? clamp01((v - thresholdNorm) / (1 - thresholdNorm || 1)) : 0;
      const alphaValue = t > 0 ? alphaFloor + (1 - alphaFloor) * Math.pow(t, alphaGamma) : 0;
      data[idx + 3] = Math.round(255 * clamp01(alphaValue));
    }
  }

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  tex.needsUpdate = true;
  tex.magFilter = THREE.LinearFilter; // Better quality
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

type Normalizer = {
  useLogZ: boolean;
  zEps: number;
  zScale: number;
  xCenter: number;
  yCenter: number;
  zCenter: number;
  xyScale: number;
  zScale3D: number;
  zMinT: number;
  zMaxT: number;
  toX: (x: number) => number;
  toY: (y: number) => number;
  toZ: (z: number) => number;
  toZ3D: (z: number) => number; // For 3D volume, Z is the third direction (gamma)
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

function buildNormalizer3D(
  X: number[][],
  Y: number[][],
  Z: number[],
  lossGrid2d: number[][],
  useLogZ: boolean
): Normalizer {
  const flatX = X.flat();
  const flatY = Y.flat();
  const flatZ = lossGrid2d.flat().filter((v) => Number.isFinite(v));

  const xMin = Math.min(...flatX);
  const xMax = Math.max(...flatX);
  const yMin = Math.min(...flatY);
  const yMax = Math.max(...flatY);
  const zMin = Math.min(...flatZ);
  const zMax = Math.max(...flatZ);
  
  // Ensure Z is a valid array
  const validZ = Array.isArray(Z) && Z.length > 0 ? Z : [-1, 1];
  const zMin3D = Math.min(...validZ);
  const zMax3D = Math.max(...validZ);

  const xCenter = (xMin + xMax) / 2;
  const yCenter = (yMin + yMax) / 2;
  const zCenter = (zMin3D + zMax3D) / 2;
  const xHalf = Math.max(Math.abs(xMax - xCenter), 1e-9);
  const yHalf = Math.max(Math.abs(yMax - yCenter), 1e-9);
  const zHalf3D = Math.max(Math.abs(zMax3D - zCenter), 1e-9);
  // For a cube-like volume, normalize α/β with their own scale, and γ with its own scale.
  // This avoids compressing γ when its numeric range is smaller than α/β.
  const xyScale = Math.max(xHalf, yHalf);
  const gammaScale = zHalf3D;

  const zEps = 1e-12;
  const zScale = 0.85; // vertical exaggeration for loss
  const zScale3D = 1.0; // scale for third dimension (γ). Use 1.0 to keep the volume box isotropic.

  const zTransform = (z: number) => (useLogZ ? Math.log10(Math.max(z, 0) + zEps) : z);
  const zMinT = zTransform(zMin);
  const zMaxT = zTransform(zMax);
  const zRangeT = zMaxT - zMinT || 1;

  const toX = (x: number) => (x - xCenter) / xyScale;
  const toY = (y: number) => (y - yCenter) / xyScale;
  const toZ = (z: number) => ((zTransform(z) - zMinT) / zRangeT) * zScale;
  const toZ3D = (z: number) => ((z - zCenter) / gammaScale) * zScale3D;

  return {
    useLogZ,
    zEps,
    zScale,
    xCenter,
    yCenter,
    zCenter,
    xyScale,
    zScale3D,
    zMinT,
    zMaxT,
    toX,
    toY,
    toZ,
    toZ3D,
    stats: { xMin, xMax, yMin, yMax, zMin, zMax, zMinT, zMaxT },
  };
}

function interpolateLoss3D(
  alpha: number,
  beta: number,
  gamma: number,
  X: number[][],
  Y: number[][],
  Z: number[],
  lossGrid3d: number[][][]
): number {
  // Improved trilinear interpolation with proper boundary handling
  const rows = X.length;
  const cols = X[0]?.length || 0;
  const depths = Z.length;

  if (rows === 0 || cols === 0 || depths === 0) {
    return 0;
  }

  // Find closest indices using distance-based search
  let minDist = Infinity;
  let bestValue = lossGrid3d[0]?.[0]?.[0] ?? 0;

  // Find closest grid point first
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < depths; k++) {
        const gridX = X[i][j];
        const gridY = Y[i][j];
        const gridZ = Z[k];
        const dist = Math.sqrt(
          (alpha - gridX) ** 2 + 
          (beta - gridY) ** 2 + 
          (gamma - gridZ) ** 2
        );
        
        if (dist < minDist) {
          minDist = dist;
          bestValue = lossGrid3d[i]?.[j]?.[k] ?? 0;
        }
      }
    }
  }

  // Try trilinear interpolation if within grid bounds
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      for (let k = 0; k < depths - 1; k++) {
        const x00 = X[i][j];
        const y00 = Y[i][j];
        const z0 = Z[k];
        const x11 = X[i + 1][j + 1];
        const y11 = Y[i + 1][j + 1];
        const z1 = Z[k + 1];

        // Check if point is within this cell
        const xMin = Math.min(x00, x11);
        const xMax = Math.max(x00, x11);
        const yMin = Math.min(y00, y11);
        const yMax = Math.max(y00, y11);
        const zMin = Math.min(z0, z1);
        const zMax = Math.max(z0, z1);

        if (alpha >= xMin && alpha <= xMax && 
            beta >= yMin && beta <= yMax &&
            gamma >= zMin && gamma <= zMax) {
          // Trilinear interpolation
          const c000 = lossGrid3d[i]?.[j]?.[k] ?? 0;
          const c001 = lossGrid3d[i]?.[j]?.[k + 1] ?? 0;
          const c010 = lossGrid3d[i]?.[j + 1]?.[k] ?? 0;
          const c011 = lossGrid3d[i]?.[j + 1]?.[k + 1] ?? 0;
          const c100 = lossGrid3d[i + 1]?.[j]?.[k] ?? 0;
          const c101 = lossGrid3d[i + 1]?.[j]?.[k + 1] ?? 0;
          const c110 = lossGrid3d[i + 1]?.[j + 1]?.[k] ?? 0;
          const c111 = lossGrid3d[i + 1]?.[j + 1]?.[k + 1] ?? 0;

          const dx = xMax > xMin ? (alpha - xMin) / (xMax - xMin) : 0.5;
          const dy = yMax > yMin ? (beta - yMin) / (yMax - yMin) : 0.5;
          const dz = zMax > zMin ? (gamma - zMin) / (zMax - zMin) : 0.5;

          const c00 = c000 * (1 - dx) + c100 * dx;
          const c01 = c001 * (1 - dx) + c101 * dx;
          const c10 = c010 * (1 - dx) + c110 * dx;
          const c11 = c011 * (1 - dx) + c111 * dx;

          const c0 = c00 * (1 - dy) + c10 * dy;
          const c1 = c01 * (1 - dy) + c11 * dy;

          return c0 * (1 - dz) + c1 * dz;
        }
      }
    }
  }

  // Fallback: return closest value
  return bestValue;
}

function TrajectoryLine3D({
  trajectory,
  X,
  Y,
  Z,
  lossGrid2d,
  lossGrid3d,
  norm,
  viewEpoch,
}: {
  trajectory: { traj_1: number[]; traj_2: number[]; traj_3?: number[]; epochs: number[] };
  X: number[][];
  Y: number[][];
  Z: number[];
  lossGrid2d: number[][];
  lossGrid3d: number[][][];
  norm: Normalizer;
  viewEpoch?: number;
}) {
  const { points, epochs } = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const epochs: number[] = [];
    const { traj_1, traj_2, traj_3 } = trajectory;

    const xMin = norm.stats.xMin;
    const xMax = norm.stats.xMax;
    const yMin = norm.stats.yMin;
    const yMax = norm.stats.yMax;
    const gMin = Array.isArray(Z) && Z.length > 0 ? Math.min(...Z) : -1;
    const gMax = Array.isArray(Z) && Z.length > 0 ? Math.max(...Z) : 1;

    let clampedAlpha = 0;
    let clampedBeta = 0;
    let clampedGamma = 0;

    // Volume view coordinate system:
    // - α (traj_1) -> X
    // - β (traj_2) -> Z (depth)
    // - γ (traj_3) -> Y (up)
    // Loss is encoded in the volume texture (color/opacity), NOT as a spatial axis.
    
    // Build array of points with their epochs for sorting
    const pointData: Array<{ point: THREE.Vector3; epoch: number; index: number }> = [];
    
    for (let i = 0; i < traj_1.length; i++) {
      const ep = trajectory.epochs[i] || i;
      if (viewEpoch !== undefined && ep > viewEpoch) continue;

      let alpha = traj_1[i];
      let beta = traj_2[i];
      
      // If traj_3 exists, use it as gamma coordinate; otherwise assume center (gamma=0)
      let gamma = 0;
      if (traj_3 && traj_3.length > i) {
        gamma = traj_3[i];
      } else if (Z && Z.length > 0) {
        // If no traj_3, use center of Z range
        gamma = (Z[0] + Z[Z.length - 1]) / 2;
      }

      // Clamp to sampled volume bounds so projections outside the computed grid
      // don't produce misleading lines far outside the cube.
      if (alpha < xMin || alpha > xMax) clampedAlpha++;
      if (beta < yMin || beta > yMax) clampedBeta++;
      if (gamma < gMin || gamma > gMax) clampedGamma++;
      alpha = Math.min(xMax, Math.max(xMin, alpha));
      beta = Math.min(yMax, Math.max(yMin, beta));
      gamma = Math.min(gMax, Math.max(gMin, gamma));
      
      // (Optional) loss available via interpolateLoss3D(...) for debugging; not used as position.
      // const loss = interpolateLoss3D(alpha, beta, gamma, X, Y, Z, lossGrid3d);

      // Map: α→X, γ→Y (up), β→Z (depth)
      pointData.push({
        point: new THREE.Vector3(
          norm.toX(alpha),
          norm.toZ3D(gamma),
          norm.toY(beta)
        ),
        epoch: ep,
        index: i,
      });
    }
    
    // Sort by epoch (low to high) to ensure correct trajectory order
    pointData.sort((a, b) => a.epoch - b.epoch);
    
    // Extract sorted points and epochs
    for (const { point, epoch } of pointData) {
      points.push(point);
      epochs.push(epoch);
    }
    
    console.log('[TrajectoryLine3D] Generated points:', {
      pointCount: points.length,
      firstPoint: points[0],
      lastPoint: points[points.length - 1],
      hasTraj3: !!traj_3,
    });
    if (clampedAlpha || clampedBeta || clampedGamma) {
      console.warn('[TrajectoryLine3D] Trajectory exceeded sampled volume bounds; points were clamped', {
        clampedAlpha,
        clampedBeta,
        clampedGamma,
        xMin,
        xMax,
        yMin,
        yMax,
        gMin,
        gMax,
      });
    }

    return { points, epochs };
  }, [trajectory, X, Y, Z, lossGrid2d, lossGrid3d, norm, viewEpoch]);

  // Sample points to show (every N-th point to avoid clutter)
  const sampledIndices = useMemo(() => {
    const total = points.length;
    if (total <= 20) return Array.from({ length: total }, (_, i) => i);
    
    const step = Math.ceil(total / 20);
    const indices: number[] = [0];
    for (let i = step; i < total - 1; i += step) {
      indices.push(i);
    }
    indices.push(total - 1);
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
      
      {/* Trajectory points as small spheres */}
      {sampledIndices.map((i) => {
        const progress = i / (points.length - 1);
        const color = new THREE.Color(
          progress < 0.5 
            ? THREE.MathUtils.lerp(0.2, 1.0, progress * 2)
            : 1.0,
          progress < 0.5 
            ? THREE.MathUtils.lerp(0.8, 1.0, progress * 2)
            : THREE.MathUtils.lerp(1.0, 0.2, (progress - 0.5) * 2),
          0.2
        );
        
        const isStart = i === 0;
        const isEnd = i === points.length - 1;
        const radius = isStart || isEnd ? 0.04 : 0.02;
        
        return (
          <mesh key={i} position={points[i]}>
            <sphereGeometry args={[radius, 16, 16]} />
            <meshStandardMaterial 
              color={color} 
              emissive={color}
              emissiveIntensity={isStart || isEnd ? 0.3 : 0.1}
            />
          </mesh>
        );
      })}
      
      {/* Start point marker */}
      {points[0] && (
        <mesh position={points[0]}>
          <sphereGeometry args={[0.05, 20, 20]} />
          <meshStandardMaterial 
            color="#00ff00" 
            emissive="#00ff00"
            emissiveIntensity={0.5}
          />
        </mesh>
      )}
      
      {/* End point marker */}
      {points[points.length - 1] && (
        <mesh position={points[points.length - 1]}>
          <sphereGeometry args={[0.05, 20, 20]} />
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

export default function LossVolumeRender3D({
  X,
  Y,
  Z,
  lossGrid2d,
  lossGrid3d,
  baselineLoss,
  trajectory,
}: {
  X: number[][];
  Y: number[][];
  Z?: number[];
  lossGrid2d: number[][];
  lossGrid3d: number[][][];
  baselineLoss?: number;
  trajectory?: { traj_1: number[]; traj_2: number[]; traj_3?: number[]; epochs: number[] };
}) {
  const { t } = useI18n();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [opacity, setOpacity] = useState(0.22);
  const [step, setStep] = useState(2);
  const [threshold, setThreshold] = useState(0.0);
  const [useLog, setUseLog] = useState(true);
  // Transfer function controls (helps visibility in dark mode)
  const [colorScale, setColorScale] = useState<'linear' | 'log'>('log');
  const [alphaGamma, setAlphaGamma] = useState(0.5);
  const [alphaFloor, setAlphaFloor] = useState(0.02);
  const [legendPos, setLegendPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingLegend = useRef(false);
  const dragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const posStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  
  // Slider panel position and drag state
  const [sliderPanelPos, setSliderPanelPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingSliderPanel = useRef(false);
  const sliderDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const sliderPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const ui = useMemo(() => {
    // Keep overlays readable in both themes (match overall app aesthetic)
    return isDark
      ? {
          panelBg: 'rgba(0,0,0,0.72)',
          panelBorder: 'rgba(255,255,255,0.22)',
          panelShadow: '0 12px 40px rgba(0,0,0,0.5)',
          text: 'rgba(255,255,255,0.95)',
          textMuted: 'rgba(255,255,255,0.75)',
          surface: 'rgba(255,255,255,0.06)',
          surfaceBorder: 'rgba(255,255,255,0.12)',
          divider: 'rgba(255,255,255,0.16)',
        }
      : {
          panelBg: 'rgba(255,255,255,0.86)',
          panelBorder: 'rgba(15,23,42,0.12)',
          panelShadow: '0 12px 40px rgba(15,23,42,0.12)',
          text: '#0f172a',
          textMuted: 'rgba(15,23,42,0.75)',
          surface: 'rgba(15,23,42,0.04)',
          surfaceBorder: 'rgba(15,23,42,0.10)',
          divider: 'rgba(15,23,42,0.12)',
        };
  }, [isDark]);

  // Compute epoch range
  const { minEpoch, maxEpoch } = useMemo(() => {
    if (!trajectory || !trajectory.epochs || trajectory.epochs.length === 0) {
      return { minEpoch: 0, maxEpoch: 0 };
    }
    // Avoid Math.min(...arr) on large arrays (can overflow call stack).
    let minEpoch = Number.POSITIVE_INFINITY;
    let maxEpoch = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < trajectory.epochs.length; i++) {
      const e = trajectory.epochs[i];
      if (!Number.isFinite(e)) continue;
      if (e < minEpoch) minEpoch = e;
      if (e > maxEpoch) maxEpoch = e;
    }
    if (!Number.isFinite(minEpoch) || !Number.isFinite(maxEpoch)) return { minEpoch: 0, maxEpoch: 0 };
    return {
      minEpoch,
      maxEpoch,
    };
  }, [trajectory]);

  const [viewEpoch, setViewEpoch] = useState<number>(maxEpoch);

  // Update viewEpoch if maxEpoch changes (e.g. data loaded)
  useMemo(() => {
    setViewEpoch(maxEpoch);
  }, [maxEpoch]);

  // Generate Z array if not provided (from lossGrid3d dimensions)
  const ZArray = useMemo(() => normalizeZAxis(Z, lossGrid3d, X, Y), [Z, lossGrid3d, X, Y]);

  const norm = useMemo(() => buildNormalizer3D(X, Y, ZArray, lossGrid2d, useLog), [X, Y, ZArray, lossGrid2d, useLog]);

  const { minV, maxV, planes, dataMinV, dataMaxV } = useMemo(() => {
    const nx = lossGrid3d.length;
    const ny = lossGrid3d[0]?.length || 0;
    const nz = lossGrid3d[0]?.[0]?.length || 0;
    const { minV: dataMinV, maxV: dataMaxV } = finiteMinMax3D(lossGrid3d);
    // Clip outliers so mid/low regions become visible (huge win on dark backgrounds).
    const { minV, maxV } = robustMinMax3D(lossGrid3d, 0.02, 0.98);
    const thresholdValue = minV + threshold * (maxV - minV);

    // Build textures for k slices and place planes at the true γ coordinate (stacked along Y)
    const planes: { y: number; tex: THREE.DataTexture }[] = [];
    const kStep = Math.max(1, step);
    for (let k = 0; k < nz; k += kStep) {
      // IMPORTANT: texture axes must align with trajectory mapping on the plane:
      // - texture X (u) should correspond to α (grid i)
      // - texture Y (v) should correspond to β (grid j)
      // sliceToTexture expects slice[v][u] == slice[betaIndex][alphaIndex]
      const slice: number[][] = Array.from({ length: ny }, (_, j) =>
        Array.from({ length: nx }, (_, i) => lossGrid3d[i][j][k] ?? 0)
      );
      const tex = sliceToTexture(slice, minV, maxV, thresholdValue, {
        scale: colorScale,
        alphaGamma,
        alphaFloor,
      });
      const gamma = ZArray[k] ?? 0;
      const y = norm.toZ3D(gamma);
      planes.push({ y, tex });
    }
    return { minV, maxV, planes, dataMinV, dataMaxV };
  }, [lossGrid3d, step, threshold, ZArray, norm, colorScale, alphaGamma, alphaFloor]);

  // Legend and slider panel drag handlers - optimized with requestAnimationFrame
  useEffect(() => {
    let rafId: number | null = null;
    
    const onMove = (e: MouseEvent) => {
      // Throttle updates with requestAnimationFrame
      if (rafId !== null) return;
      
      rafId = requestAnimationFrame(() => {
        rafId = null;
        
        if (isDraggingLegend.current) {
          const dx = e.clientX - dragStart.current.x;
          const dy = e.clientY - dragStart.current.y;
          setLegendPos({ x: posStart.current.x + dx, y: posStart.current.y + dy });
        }
        if (isDraggingSliderPanel.current) {
          const dx = e.clientX - sliderDragStart.current.x;
          const dy = e.clientY - sliderDragStart.current.y;
          setSliderPanelPos({ x: sliderPosStart.current.x + dx, y: sliderPosStart.current.y + dy });
        }
      });
    };
    
    const onUp = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (isDraggingLegend.current) isDraggingLegend.current = false;
      if (isDraggingSliderPanel.current) isDraggingSliderPanel.current = false;
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

  const startSliderPanelDrag = (e: React.MouseEvent) => {
    isDraggingSliderPanel.current = true;
    sliderDragStart.current = { x: e.clientX, y: e.clientY };
    sliderPosStart.current = sliderPanelPos;
    e.preventDefault();
  };

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas>
        <color attach="background" args={[isDark ? '#1a1a1a' : '#f1f5f9']} />
        <PerspectiveCamera makeDefault position={[2.0, 2.0, 1.6]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        {/* Each plane is an (α,β) slice at a specific γ (stacked along Y). */}
        {planes.map((p, idx) => (
          // Rotate plane so it spans (α on X) and (β on Z) with the same sign convention
          // as the trajectory mapping (z = norm.toY(beta)).
          <mesh key={idx} position={[0, p.y, 0]} rotation={[Math.PI / 2, 0, 0]}>
            {/* Use a 2×2 plane so X/Z span roughly [-1, 1] in normalized space (cube-like volume) */}
            <planeGeometry args={[2.0, 2.0]} />
            <meshBasicMaterial
              map={p.tex}
              transparent
              opacity={opacity}
              depthWrite={false}
              // Additive blending makes faint low values accumulate instead of disappearing in dark mode.
              blending={isDark ? THREE.AdditiveBlending : THREE.NormalBlending}
              side={THREE.DoubleSide}
            />
          </mesh>
        ))}
        {trajectory && (
          <TrajectoryLine3D
            trajectory={trajectory}
            X={X}
            Y={Y}
            Z={ZArray}
            lossGrid2d={lossGrid2d}
            lossGrid3d={lossGrid3d}
            norm={norm}
            viewEpoch={viewEpoch}
          />
        )}
        <OrbitControls />
      </Canvas>

      {/* Colorbar Legend (top-right, same style as 2D/3D Surface) */}
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
          <span>Loss Volume (3D)</span>
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
                    const c = getViridisColor(t);
                    const rgb = `rgb(${c.r},${c.g},${c.b})`;
                    const stopPercent = i === 0 ? '0%' : i === steps ? '100%' : `${(i / steps * 100).toFixed(2)}%`;
                    colors.push(`${rgb} ${stopPercent}`);
                  }
                  return colors.join(', ');
                })()})`,
                boxShadow: isDark
                  ? '0 4px 12px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,255,255,0.08)'
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
              Hover for hint
            </div>
          </div>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 160 }}>
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(255,100,100,0.15)' : 'rgba(239,68,68,0.10)',
              border: isDark ? '1px solid rgba(255,100,100,0.3)' : '1px solid rgba(239,68,68,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>High Loss</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#ff6b6b' }}>{fmtLoss(maxV)}</div>
            </div>
            
            <div style={{ 
              textAlign: 'center',
              padding: '6px 0',
              fontSize: 10,
              opacity: isDark ? 0.7 : 0.8,
              fontStyle: 'italic',
            }}>
              ↓ Lower is better ↓
            </div>
            
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(100,255,150,0.15)' : 'rgba(16,185,129,0.10)',
              border: isDark ? '1px solid rgba(100,255,150,0.3)' : '1px solid rgba(16,185,129,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Low Loss</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#51cf66' }}>{fmtLoss(minV)}</div>
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

          <div style={{ fontSize: 10, opacity: 0.7, lineHeight: 1.35 }}>
            <div>
              Color range (p2–p98): <span style={{ fontFamily: 'monospace' }}>{fmtLoss(minV)} → {fmtLoss(maxV)}</span>
            </div>
            <div>
              {t.dataMinMax} <span style={{ fontFamily: 'monospace' }}>{fmtLoss(dataMinV)} → {fmtLoss(dataMaxV)}</span>
            </div>
          </div>

          {/* Match other views: Scale toggle lives in the legend */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: ui.surface,
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <span style={{ opacity: 0.9, fontWeight: 600 }}>Scale</span>
            <button
              type="button"
              onClick={() => setColorScale((s) => (s === 'log' ? 'linear' : 'log'))}
              style={{
                border: `2px solid ${
                  colorScale === 'log'
                    ? 'rgba(251, 191, 36, 0.55)'
                    : isDark
                      ? 'rgba(255,255,255,0.30)'
                      : 'rgba(15,23,42,0.18)'
                }`,
                background:
                  colorScale === 'log'
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
              onMouseDown={(e) => e.stopPropagation()}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background =
                  colorScale === 'log'
                    ? 'rgba(251, 191, 36, 0.28)'
                    : isDark
                      ? 'rgba(255,255,255,0.12)'
                      : 'rgba(15,23,42,0.07)')
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background =
                  colorScale === 'log'
                    ? 'rgba(251, 191, 36, 0.20)'
                    : isDark
                      ? 'rgba(255,255,255,0.08)'
                      : 'rgba(15,23,42,0.05)')
              }
            >
              {colorScale === 'log' ? `📊 ${t.log}` : `📈 ${t.linear}`}
            </button>
          </div>
          
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

          {trajectory && maxEpoch > 0 && (
            <div 
              style={{ 
                display: 'flex', 
                gap: 12, 
                alignItems: 'center', 
                fontSize: 12,
                padding: '8px 10px',
                borderRadius: 8,
                background: ui.surface,
              }}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <span style={{ opacity: 0.9, fontWeight: 600 }}>Epoch:</span>
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
          💡 <strong>Tip:</strong> Drag to rotate, scroll to zoom, adjust opacity/threshold below
        </div>
      </div>

      {/* Slider controls panel (bottom-right, draggable, same width as legend) */}
      <div
        style={{
          position: 'absolute',
          right: 16,
          bottom: 16,
          transform: `translate(${sliderPanelPos.x}px, ${sliderPanelPos.y}px)`,
          display: 'grid',
          gap: 10,
          padding: '14px 16px',
          borderRadius: 16,
          border: `1px solid ${ui.panelBorder}`,
          background: ui.panelBg,
          backdropFilter: 'blur(10px)',
          color: ui.text,
          fontSize: 13,
          width: 300,
          boxSizing: 'border-box',
          boxShadow: ui.panelShadow,
          zIndex: 10,
          overflow: 'hidden',
        }}
      >
        <div 
          style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            gap: 12, 
            alignItems: 'center', 
            marginBottom: 4,
            cursor: isDraggingSliderPanel.current ? 'grabbing' : 'grab',
            minWidth: 0,
          }}
          onMouseDown={startSliderPanelDrag}
        >
          <div style={{ fontWeight: 800, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}>{t.view3DVolume}</div>
          <div style={{ opacity: isDark ? 0.85 : 0.8, fontSize: 11, fontFamily: 'monospace', flexShrink: 0 }}>
            {planes.length} {t.slices}
          </div>
        </div>
        <div style={{ fontSize: 11, opacity: isDark ? 0.75 : 0.78, marginBottom: 8, lineHeight: 1.4, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
          {t.volumeRenderingDesc}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '85px 1fr 55px', gap: 8, alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.opacity}</div>
          <input
            type="range"
            min={0.05}
            max={0.5}
            step={0.01}
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            onMouseDown={(e) => e.stopPropagation()}
            style={{ accentColor: '#fbbf24', minWidth: 0 }}
          />
          <div style={{ textAlign: 'right', opacity: 0.9, fontFamily: 'monospace', fontSize: 11 }}>{opacity.toFixed(2)}</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '85px 1fr 55px', gap: 8, alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.step}</div>
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={step}
            onChange={(e) => setStep(parseInt(e.target.value, 10))}
            onMouseDown={(e) => e.stopPropagation()}
            style={{ accentColor: '#fbbf24', minWidth: 0 }}
          />
          <div style={{ textAlign: 'right', opacity: 0.9, fontFamily: 'monospace', fontSize: 11 }}>{step}</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '85px 1fr 55px', gap: 8, alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.threshold}</div>
          <input
            type="range"
            min={0}
            max={0.8}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            onMouseDown={(e) => e.stopPropagation()}
            style={{ accentColor: '#fbbf24', minWidth: 0 }}
          />
          <div style={{ textAlign: 'right', opacity: 0.9, fontFamily: 'monospace', fontSize: 11 }}>{threshold.toFixed(2)}</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '85px 1fr 55px', gap: 8, alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.alphaCurve}</div>
          <input
            type="range"
            min={0.2}
            max={1.2}
            step={0.01}
            value={alphaGamma}
            onChange={(e) => setAlphaGamma(parseFloat(e.target.value))}
            onMouseDown={(e) => e.stopPropagation()}
            style={{ accentColor: '#fbbf24', minWidth: 0 }}
          />
          <div style={{ textAlign: 'right', opacity: 0.9, fontFamily: 'monospace', fontSize: 11 }}>{alphaGamma.toFixed(2)}</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '85px 1fr 55px', gap: 8, alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.alphaFloor}</div>
          <input
            type="range"
            min={0}
            max={0.12}
            step={0.005}
            value={alphaFloor}
            onChange={(e) => setAlphaFloor(parseFloat(e.target.value))}
            onMouseDown={(e) => e.stopPropagation()}
            style={{ accentColor: '#fbbf24', minWidth: 0 }}
          />
          <div style={{ textAlign: 'right', opacity: 0.9, fontFamily: 'monospace', fontSize: 11 }}>{alphaFloor.toFixed(3)}</div>
        </div>

        <div style={{ fontSize: 10, opacity: 0.6, marginTop: 4, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
          Filter low-loss regions (0 = show all, higher = only high-loss)
        </div>
      </div>
    </div>
  );
}



