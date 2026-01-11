/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useMemo, useState, useEffect } from 'react';
import LossLandscape2D from './LossLandscape2D';

// Normalize Z input from backend:
// - Preferred shape: 1D array [gamma_k]
// - Also support legacy / volume-exported shape: 3D grid Z[i][j][k]
function normalizeZAxis(Z: any, lossGrid3d: number[][][]): number[] {
  const nz = lossGrid3d?.[0]?.[0]?.length || 0;
  if (nz <= 0) return [];

  if (!Array.isArray(Z)) return [];

  // 1D numeric or numeric-like array
  const first = Z[0];
  if (typeof first === 'number' || typeof first === 'string') {
    const arr = (Z as any[])
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v));
    if (arr.length === nz) return arr;
    if (arr.length > 0) {
      // fallback: resample/trim to nz points
      const step = Math.max(1, Math.floor(arr.length / nz));
      const picked: number[] = [];
      for (let i = 0; i < arr.length && picked.length < nz; i += step) {
        picked.push(arr[i]);
      }
      while (picked.length < nz) picked.push(arr[arr.length - 1]);
      return picked;
    }
    return [];
  }

  // 3D grid case: assume shape [nx][ny][nz] and take the k-axis from the first cell
  if (Array.isArray(first)) {
    const zLine = (Z[0]?.[0] ?? []) as any[];
    const arr = zLine.map((v) => Number(v)).filter((v) => Number.isFinite(v));
    if (arr.length === nz) return arr;
  }

  return [];
}

export default function LossVolumeSlice2D({
  X,
  Y,
  Z,
  lossGrid3d,
  trajectory,
  k: kProp,
  onKChange: _onKChange,
  xLabel,
  planeLabel,
}: {
  X: number[][];
  Y: number[][];
  Z?: number[];
  lossGrid3d: number[][][];
  trajectory?: { traj_1: number[]; traj_2: number[]; traj_3?: number[]; epochs: number[] };
  k?: number;
  onKChange?: (k: number) => void;
  xLabel?: string;
  planeLabel?: string;
}) {
  const [kInternal, setKInternal] = useState(0);

  const kMax = Math.max(0, lossGrid3d?.[0]?.[0]?.length - 1 || 0);
  const k = Math.max(0, Math.min(kProp ?? kInternal, kMax));

  // Sync internal state when controlled prop changes
  useEffect(() => {
    if (kProp === undefined) return;
    const clamped = Math.max(0, Math.min(kProp, kMax));
    setKInternal(clamped);
  }, [kProp, kMax]);

  // Compute epoch range
  const { maxEpoch } = useMemo(() => {
    if (!trajectory || !trajectory.epochs || trajectory.epochs.length === 0) {
      return { maxEpoch: 0 };
    }
    return {
      maxEpoch: Math.max(...trajectory.epochs),
    };
  }, [trajectory]);

  const [viewEpoch, setViewEpoch] = useState<number>(maxEpoch);

  // Update viewEpoch if maxEpoch changes
  useEffect(() => {
    setViewEpoch(maxEpoch);
  }, [maxEpoch]);

  // Depth (gamma axis) is the 3rd dimension of lossGrid3d: lossGrid3d[x][y][k]
  const nz = lossGrid3d?.[0]?.[0]?.length || 0;

  // IMPORTANT: Z and traj_3 are in the same units as the perturbation coefficients (±step_size in parameter space).
  // We must NOT invent a normalized Z unless Z is missing. If Z is missing, the only “best effort” we can do
  // is to derive a grid from traj_3 range; otherwise we cannot correctly filter trajectory by slice.
  const zVals = useMemo(() => {
    // Try to normalize Z from backend (supports 1D or 3D grid)
    const normalized = normalizeZAxis(Z, lossGrid3d);
    if (normalized.length === nz && nz > 0) return normalized;

    if (trajectory?.traj_3 && trajectory.traj_3.length > 0 && nz > 0) {
      const minG = Math.min(...trajectory.traj_3);
      const maxG = Math.max(...trajectory.traj_3);
      console.warn('[LossVolumeSlice2D] Z missing/mismatched; deriving z grid from traj_3 range', {
        nz,
        minG,
        maxG,
        ZShape: Array.isArray(Z) ? 'array' : typeof Z,
      });
      // Linear grid across observed gamma range. This is approximate but at least in the right units.
      return Array.from({ length: nz }, (_, i) => minG + (i / Math.max(1, nz - 1)) * (maxG - minG));
    }

    // Without Z and without traj_3, we cannot define gamma. Return empty.
    console.warn('[LossVolumeSlice2D] Z missing and no traj_3; gamma slicing disabled', {
      nz,
      ZShape: Array.isArray(Z) ? 'array' : typeof Z,
      hasTraj3: !!trajectory?.traj_3,
    });
    return [] as number[];
  }, [Z, nz, trajectory?.traj_3, lossGrid3d]);
  
  const slice = useMemo(() => {
    const rows = lossGrid3d.length;
    const cols = lossGrid3d[0]?.length || 0;
    const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) out[i][j] = lossGrid3d[i][j][k] ?? 0;
    }
    return out;
  }, [lossGrid3d, k]);

  // Filter trajectory based on viewEpoch (for projection)
  const epochFilteredTrajectory = useMemo(() => {
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

  // Calculate slice parameters
  const { currentGamma, sliceThreshold } = useMemo(() => {
    if (!zVals || zVals.length !== nz || zVals.length === 0) {
      return { currentGamma: 0, sliceThreshold: 0 };
    }
    const currentGamma = zVals[Math.max(0, Math.min(k, zVals.length - 1))];

    // Slice thickness: half the spacing to neighboring slices (local, not global).
    let halfThickness = 0;
    if (zVals.length <= 1) {
      halfThickness = 0;
    } else if (k <= 0) {
      halfThickness = Math.abs(zVals[1] - zVals[0]) / 2;
    } else if (k >= zVals.length - 1) {
      halfThickness = Math.abs(zVals[zVals.length - 1] - zVals[zVals.length - 2]) / 2;
    } else {
      const left = Math.abs(zVals[k] - zVals[k - 1]);
      const right = Math.abs(zVals[k + 1] - zVals[k]);
      halfThickness = Math.min(left, right) / 2;
    }

    const eps = 1e-12;
    return { currentGamma, sliceThreshold: halfThickness + eps };
  }, [zVals, k, nz]);

  // Filter trajectory points that are close to current slice AND within viewEpoch
  const filteredTrajectory = useMemo(() => {
    if (!trajectory || !trajectory.traj_1 || trajectory.traj_1.length === 0) return undefined;

    // If we don't have gamma coordinate in the trajectory, this is effectively a 2D trajectory.
    // In that case, show it on every slice (it lies in the α-β plane).
    if (!trajectory.traj_3 || trajectory.traj_3.length === 0) return trajectory;

    // If we don't have a valid gamma grid, we cannot slice properly.
    if (!zVals || zVals.length !== nz || zVals.length === 0) return trajectory;

    console.log('[LossVolumeSlice2D] Slice trajectory filter:', {
      k,
      nz,
      currentGamma,
      sliceThreshold,
      trajLen: trajectory.traj_1.length,
      traj3Len: trajectory.traj_3.length,
      zMin: zVals[0],
      zMax: zVals[zVals.length - 1],
      traj3Min: Math.min(...trajectory.traj_3),
      traj3Max: Math.max(...trajectory.traj_3),
    });

    const filtered: { traj_1: number[]; traj_2: number[]; traj_3: number[]; epochs: number[] } = {
      traj_1: [],
      traj_2: [],
      traj_3: [],
      epochs: [],
    };

    // Strict slice membership: only points whose gamma lies within the slice slab.
    const n = Math.min(trajectory.traj_1.length, trajectory.traj_2.length, trajectory.traj_3.length);
    for (let i = 0; i < n; i++) {
      const ep = trajectory.epochs[i] ?? i;
      if (ep > viewEpoch) continue;

      const gamma = trajectory.traj_3[i];
      if (!Number.isFinite(gamma)) continue;
      if (Math.abs(gamma - currentGamma) <= sliceThreshold) {
        filtered.traj_1.push(trajectory.traj_1[i]);
        filtered.traj_2.push(trajectory.traj_2[i]);
        filtered.traj_3.push(gamma);
        filtered.epochs.push(ep);
      }
    }

    console.log('[LossVolumeSlice2D] Filtered trajectory:', {
      originalLength: n,
      filteredLength: filtered.traj_1.length,
      currentGamma,
      hasTraj3: !!trajectory.traj_3,
    });

    return filtered.traj_1.length > 0 ? filtered : undefined;
  }, [trajectory, zVals, k, nz, viewEpoch, currentGamma, sliceThreshold]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Slice view: show full α–β projection of the trajectory, and highlight points in the current γ slab */}
      <LossLandscape2D
        X={X}
        Y={Y}
        lossGrid={slice}
        trajectory={epochFilteredTrajectory} // projection (α,β) — independent of γ
        trajectoryHighlight={filteredTrajectory} // highlighted in-slab points
        sliceGamma={currentGamma}
        sliceThreshold={sliceThreshold}
        xLabel={xLabel}
        planeLabel={planeLabel}
      />
    </div>
  );
}


