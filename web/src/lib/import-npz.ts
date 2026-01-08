/**
 * Unified helper function for importing landscape data files
 * Only supports the standard format
 */

import { saveLossLandscape, LossLandscapeData } from '@/lib/db';
import fs from 'fs';
import path from 'path';

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
  trajectory_losses?: number[];
  trajectory_val_losses?: number[];
  metadata?: any; // Training metadata from JSON file
  trajectory_data?: {
    traj_1?: number[];
    traj_2?: number[];
    traj_3?: number[];
    epochs?: number[];
    losses?: number[];
    val_losses?: number[];
  };
}

/**
 * Read export metadata from .export.meta.json file if it exists
 * @param dataFilePath Path to the data file (.json)
 * @returns Export metadata object or null if file doesn't exist
 */
export function readExportMetadata(dataFilePath: string): any | null {
  try {
    // For .json files, metadata is at same path with .export.meta.json extension
    // e.g., complete_example.json -> complete_example.export.meta.json
    let metadataPath: string;
    
    if (dataFilePath.endsWith('.json')) {
      // Direct JSON file: replace .json with .export.meta.json
      metadataPath = dataFilePath.replace(/\.json$/, '.export.meta.json');
    } else {
      return null;
    }
    
    if (fs.existsSync(metadataPath)) {
      const metadataContent = fs.readFileSync(metadataPath, 'utf-8');
      return JSON.parse(metadataContent);
    }
  } catch (e) {
    // If metadata file is corrupted or can't be read, ignore it
    console.warn(`Failed to read export metadata from ${dataFilePath}:`, e);
  }
  
  return null;
}

/**
 * Read metadata and trajectory data from JSON file if it exists
 * @param dataFilePath Path to the data file (.json)
 * @returns Object with metadata, trajectory_losses, and trajectory_val_losses, or null
 */
export function readJsonFileData(dataFilePath: string): { metadata?: any; trajectory_losses?: number[]; trajectory_val_losses?: number[] } | null {
  try {
    let jsonFilePath: string | undefined;
    
    if (dataFilePath.endsWith('.json')) {
      jsonFilePath = dataFilePath;
    } else {
      return null;
    }
    
    if (jsonFilePath && fs.existsSync(jsonFilePath)) {
      try {
        const jsonContent = fs.readFileSync(jsonFilePath, 'utf-8');
        // Sanitize NaN/Infinity for JSON parsing
        const sanitizedContent = jsonContent
          .replace(/\bNaN\b/g, 'null')
          .replace(/\bInfinity\b/g, 'null')
          .replace(/\b-Infinity\b/g, 'null');
        const jsonData = JSON.parse(sanitizedContent);
        
        const result: { metadata?: any; trajectory_losses?: number[]; trajectory_val_losses?: number[] } = {};
        
        // Extract metadata
        if (jsonData.metadata) {
          result.metadata = jsonData.metadata;
        }
        
        // Extract trajectory losses
        if (jsonData.trajectory_data) {
          if (jsonData.trajectory_data.losses) {
            result.trajectory_losses = jsonData.trajectory_data.losses;
          }
          if (jsonData.trajectory_data.val_losses) {
            result.trajectory_val_losses = jsonData.trajectory_data.val_losses;
          }
        }
        
        // Also check top-level fields
        if (jsonData.trajectory_losses) {
          result.trajectory_losses = jsonData.trajectory_losses;
        }
        if (jsonData.trajectory_val_losses) {
          result.trajectory_val_losses = jsonData.trajectory_val_losses;
        }
        
        return Object.keys(result).length > 0 ? result : null;
      } catch (e) {
        console.warn(`Failed to parse JSON file ${jsonFilePath}:`, e);
      }
    }
  } catch (e) {
    console.warn(`Failed to read JSON file data from ${dataFilePath}:`, e);
  }
  
  return null;
}

/**
 * Normalize and import npz data to database
 * Only supports standard format - no fallbacks or compatibility modes
 */
export async function importNpzData(
  data: NpzData,
  sourceLabel: string,
  sourceFilePath?: string,
  exportMetadata?: any,
  runDir?: string,
  importSource?: string,
  importFilename?: string
): Promise<number> {
  // Check if we have at least one type of data: 2D surface, 3D volume, or trajectory
  const has2DData = Array.isArray(data.X) && Array.isArray(data.Y) && Array.isArray(data.loss_grid_2d) &&
                    data.X.length > 0 && data.Y.length > 0 && data.loss_grid_2d.length > 0;
  const has3DData = Array.isArray(data.loss_grid_3d) && data.loss_grid_3d.length > 0;
  const hasTrajectoryData = data.trajectory_data && 
                            Array.isArray(data.trajectory_data.epochs) && 
                            data.trajectory_data.epochs.length > 0;

  if (!has2DData && !has3DData && !hasTrajectoryData) {
    throw new Error('Landscape file must contain at least one of: 2D surface data (X, Y, loss_grid_2d), 3D volume data (loss_grid_3d), or trajectory data');
  }

  // Ensure X, Y, loss_grid_2d exist (create empty arrays if needed for 3D-only or trajectory-only data)
  const X = Array.isArray(data.X) ? data.X : [];
  const Y = Array.isArray(data.Y) ? data.Y : [];
  const loss_grid_2d = Array.isArray(data.loss_grid_2d) ? data.loss_grid_2d : [];

  // Calculate grid_size from data if not provided
  // Try multiple sources: explicit grid_size, X.length, loss_grid_2d.length, or 3D data
  let gridSize = data.grid_size;
  if (!gridSize || gridSize === 0) {
    if (X.length > 0) {
      // X is 2D array, grid_size is the number of rows
      gridSize = X.length;
    } else if (loss_grid_2d.length > 0) {
      // Fallback to loss_grid_2d dimensions
      gridSize = loss_grid_2d.length;
    } else if (Y.length > 0) {
      // Fallback to Y dimensions
      gridSize = Y.length;
    } else if (has3DData && Array.isArray(data.loss_grid_3d) && data.loss_grid_3d.length > 0) {
      // Use 3D grid size if available
      gridSize = data.loss_grid_3d.length;
    } else {
      // Default to 0 if no data available (shouldn't happen due to check above)
      gridSize = 0;
    }
  }
  
  if (!gridSize || gridSize === 0) {
    // If we have trajectory data but no grid data, set a default grid_size
    if (hasTrajectoryData && !has2DData && !has3DData) {
      gridSize = 1; // Minimal grid size for trajectory-only data
    } else {
      throw new Error('Cannot determine grid_size from data. Please ensure the landscape file contains valid 2D or 3D surface data.');
    }
  }

  // Prepare trajectory data if available
  // Use data.trajectory_data if available, otherwise construct from individual fields
  let trajectoryData = data.trajectory_data;
  if (!trajectoryData && data.trajectory_1 && data.trajectory_2) {
    trajectoryData = {
      traj_1: data.trajectory_1,
      traj_2: data.trajectory_2,
      traj_3: data.trajectory_3,
      epochs: data.trajectory_epochs || [],
    };
  }
  
  // Add losses if available from data
  if (trajectoryData) {
    if (data.trajectory_losses) {
      trajectoryData.losses = data.trajectory_losses;
    }
    if (data.trajectory_val_losses) {
      trajectoryData.val_losses = data.trajectory_val_losses;
    }
  }

  // Read JSON file data (metadata and trajectory losses) if source file path is available
  let jsonFileData: { metadata?: any; trajectory_losses?: number[]; trajectory_val_losses?: number[] } | null = null;
  if (sourceFilePath) {
    jsonFileData = readJsonFileData(sourceFilePath);
    
    // Add losses to trajectory_data if found in JSON file
    if (jsonFileData && trajectoryData) {
      if (jsonFileData.trajectory_losses) {
        trajectoryData.losses = jsonFileData.trajectory_losses;
      }
      if (jsonFileData.trajectory_val_losses) {
        trajectoryData.val_losses = jsonFileData.trajectory_val_losses;
      }
    }
  }

  // Read export metadata (.export.meta.json) if not provided and source file path is available
  let exportMeta = exportMetadata;
  if (!exportMeta && sourceFilePath) {
    exportMeta = readExportMetadata(sourceFilePath);
  }

  // Determine which metadata to store: prefer JSON file metadata over export metadata
  // Store JSON file metadata in export_metadata field (it's the actual training metadata)
  const metadataToStore = data.metadata || jsonFileData?.metadata || exportMeta;

  // Build import data - only standard format
  const importData: LossLandscapeData = {
    config_path: sourceLabel,
    run_dir: runDir || undefined,
    mode: has3DData ? '3d' : (has2DData ? '2d' : (data.mode || '2d')),
    direction: data.direction || 'unknown',
    grid_size: gridSize,
    X: X,
    Y: Y,
    Z: data.Z,
    loss_grid_2d: loss_grid_2d,
    loss_grid_3d: data.loss_grid_3d,
    baseline_loss: data.baseline_loss || 0,
    trajectory_data: trajectoryData,
    export_metadata: metadataToStore || undefined,
    import_source: importSource || undefined,
    import_filename: importFilename || (sourceFilePath ? path.basename(sourceFilePath) : undefined),
    imported_at: new Date(),
  };

  return await saveLossLandscape(importData);
}

