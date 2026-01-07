/**
 * Unified helper function for importing .npz files
 * Only supports the standard format from npzToJson
 */

import { saveLossLandscape, LossLandscapeData } from '@/lib/db';

export interface NpzData {
  X: number[][];
  Y: number[][];
  Z?: number[];
  loss_grid_2d: number[][];
  loss_grid_3d?: number[][][];
  baseline_loss?: number;
  mode?: string;
  direction?: string;
  grid_size?: number;
  trajectory_1?: number[];
  trajectory_2?: number[];
  trajectory_3?: number[];
  trajectory_epochs?: number[];
}

/**
 * Normalize and import npz data to database
 * Only supports standard format - no fallbacks or compatibility modes
 */
export async function importNpzData(
  data: NpzData,
  sourceLabel: string
): Promise<number> {
  // Validate required fields
  if (!data.X || !data.Y || !data.loss_grid_2d) {
    throw new Error('Missing required fields: X, Y, loss_grid_2d');
  }

  // Calculate grid_size from data if not provided
  const gridSize = data.grid_size || (Array.isArray(data.X) ? data.X.length : 0);
  if (gridSize === 0) {
    throw new Error('Cannot determine grid_size from data');
  }

  // Prepare trajectory data if available
  const trajectoryData =
    data.trajectory_1 && data.trajectory_2
      ? {
          traj_1: data.trajectory_1,
          traj_2: data.trajectory_2,
          traj_3: data.trajectory_3,
          epochs: data.trajectory_epochs || [],
        }
      : undefined;

  // Build import data - only standard format
  const importData: LossLandscapeData = {
    config_path: sourceLabel,
    run_dir: undefined,
    mode: data.mode || '2d',
    direction: data.direction || 'unknown',
    grid_size: gridSize,
    X: data.X,
    Y: data.Y,
    Z: data.Z,
    loss_grid_2d: data.loss_grid_2d,
    loss_grid_3d: data.loss_grid_3d,
    baseline_loss: data.baseline_loss || 0,
    trajectory_data: trajectoryData,
  };

  return await saveLossLandscape(importData);
}

