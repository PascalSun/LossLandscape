/**
 * Validate landscape JSON file format
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Validate landscape JSON data structure
 */
export function validateLandscapeData(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check required fields
  if (!data) {
    errors.push('Data is null or undefined');
    return { valid: false, errors, warnings };
  }

  // Check if we have at least one type of data: 2D surface, 3D volume, or trajectory
  const has2DData = Array.isArray(data.X) && Array.isArray(data.Y) && Array.isArray(data.loss_grid_2d) &&
                    data.X.length > 0 && data.Y.length > 0 && data.loss_grid_2d.length > 0;
  
  // Check for 3D data - loss_grid_3d should be a 3D array
  let has3DData = false;
  if (Array.isArray(data.loss_grid_3d) && data.loss_grid_3d.length > 0) {
    // Check if it's actually a 3D array (nested arrays)
    if (Array.isArray(data.loss_grid_3d[0]) && Array.isArray(data.loss_grid_3d[0][0])) {
      has3DData = true;
    }
  }
  
  // Also check for Z array as indicator of 3D data
  if (!has3DData && Array.isArray(data.Z) && data.Z.length > 0) {
    has3DData = true;
  }
  
  // Check for trajectory data
  const hasTrajectoryData = data.trajectory_data && 
                            ((Array.isArray(data.trajectory_data.epochs) && data.trajectory_data.epochs.length > 0) ||
                            (Array.isArray(data.trajectory_data.traj_1) && data.trajectory_data.traj_1.length > 0) ||
                            (Array.isArray(data.trajectory_data.traj_2) && data.trajectory_data.traj_2.length > 0));

  if (!has2DData && !has3DData && !hasTrajectoryData) {
    // Provide detailed debug information
    const debugInfo = {
      hasX: Array.isArray(data.X),
      xLength: Array.isArray(data.X) ? data.X.length : 0,
      hasY: Array.isArray(data.Y),
      yLength: Array.isArray(data.Y) ? data.Y.length : 0,
      hasLossGrid2d: Array.isArray(data.loss_grid_2d),
      lossGrid2dLength: Array.isArray(data.loss_grid_2d) ? data.loss_grid_2d.length : 0,
      hasLossGrid3d: Array.isArray(data.loss_grid_3d),
      lossGrid3dLength: Array.isArray(data.loss_grid_3d) ? data.loss_grid_3d.length : 0,
      hasZ: Array.isArray(data.Z),
      zLength: Array.isArray(data.Z) ? data.Z.length : 0,
      hasTrajectoryData: !!data.trajectory_data,
      trajectoryEpochsLength: data.trajectory_data && Array.isArray(data.trajectory_data.epochs) 
        ? data.trajectory_data.epochs.length : 0,
      allKeys: Object.keys(data || {}),
    };
    
    errors.push(
      'Landscape file must contain at least one of: 2D surface data (X, Y, loss_grid_2d), 3D volume data (loss_grid_3d), or trajectory data. ' +
      `Debug info: ${JSON.stringify(debugInfo)}`
    );
    return { valid: false, errors, warnings };
  }

  // Validate 2D data if present
  if (Array.isArray(data.X) || Array.isArray(data.Y) || Array.isArray(data.loss_grid_2d)) {
    // If any 2D field exists, all should exist and be valid
    if (!Array.isArray(data.X)) {
      errors.push('Missing or invalid field: X (must be a 2D array)');
    } else if (data.X.length === 0) {
      // Empty is OK if we have other data types
      if (!has3DData && !hasTrajectoryData) {
        errors.push('Field X is empty and no other data types found');
      } else {
        warnings.push('Field X is empty (but 3D or trajectory data exists)');
      }
    } else if (!Array.isArray(data.X[0])) {
      errors.push('Field X must be a 2D array');
    }

    if (!Array.isArray(data.Y)) {
      errors.push('Missing or invalid field: Y (must be a 2D array)');
    } else if (data.Y.length === 0) {
      if (!has3DData && !hasTrajectoryData) {
        errors.push('Field Y is empty and no other data types found');
      } else {
        warnings.push('Field Y is empty (but 3D or trajectory data exists)');
      }
    } else if (!Array.isArray(data.Y[0])) {
      errors.push('Field Y must be a 2D array');
    }

    if (!Array.isArray(data.loss_grid_2d)) {
      errors.push('Missing or invalid field: loss_grid_2d (must be a 2D array)');
    } else if (data.loss_grid_2d.length === 0) {
      if (!has3DData && !hasTrajectoryData) {
        errors.push('Field loss_grid_2d is empty and no other data types found');
      } else {
        warnings.push('Field loss_grid_2d is empty (but 3D or trajectory data exists)');
      }
    } else if (!Array.isArray(data.loss_grid_2d[0])) {
      errors.push('Field loss_grid_2d must be a 2D array');
    }
  }

  // Validate dimensions match
  if (Array.isArray(data.X) && Array.isArray(data.Y) && Array.isArray(data.loss_grid_2d)) {
    const xRows = data.X.length;
    const yRows = data.Y.length;
    const lossRows = data.loss_grid_2d.length;

    if (xRows !== yRows || xRows !== lossRows) {
      errors.push(`Dimension mismatch: X has ${xRows} rows, Y has ${yRows} rows, loss_grid_2d has ${lossRows} rows`);
    }

    // Check column dimensions
    if (xRows > 0 && yRows > 0 && lossRows > 0) {
      const xCols = data.X[0].length;
      const yCols = data.Y[0].length;
      const lossCols = data.loss_grid_2d[0].length;

      if (xCols !== yCols || xCols !== lossCols) {
        errors.push(`Column dimension mismatch: X has ${xCols} cols, Y has ${yCols} cols, loss_grid_2d has ${lossCols} cols`);
      }
    }
  }

  // Optional: 3D data
  if (data.loss_grid_3d) {
    if (!Array.isArray(data.loss_grid_3d)) {
      errors.push('Field loss_grid_3d must be a 3D array');
    } else if (data.loss_grid_3d.length === 0) {
      warnings.push('Field loss_grid_3d is empty');
    } else if (!Array.isArray(data.loss_grid_3d[0])) {
      errors.push('Field loss_grid_3d must be a 3D array');
    } else if (data.loss_grid_3d[0].length > 0 && !Array.isArray(data.loss_grid_3d[0][0])) {
      errors.push('Field loss_grid_3d must be a 3D array');
    }

    // If 3D data exists, Z should also exist
    if (!data.Z) {
      warnings.push('3D data (loss_grid_3d) exists but Z is missing');
    } else if (!Array.isArray(data.Z)) {
      errors.push('Field Z must be an array');
    }
  }

  // Optional: trajectory_data
  if (data.trajectory_data) {
    if (typeof data.trajectory_data !== 'object') {
      errors.push('Field trajectory_data must be an object');
    } else {
      if (data.trajectory_data.traj_1 !== undefined && !Array.isArray(data.trajectory_data.traj_1)) {
        warnings.push('trajectory_data.traj_1 is missing or invalid');
      }
      if (data.trajectory_data.traj_2 !== undefined && !Array.isArray(data.trajectory_data.traj_2)) {
        warnings.push('trajectory_data.traj_2 is missing or invalid');
      }
      if (data.trajectory_data.epochs !== undefined && !Array.isArray(data.trajectory_data.epochs)) {
        warnings.push('trajectory_data.epochs is missing or invalid');
      }
    }
  }

  // Optional: metadata
  if (data.metadata && typeof data.metadata !== 'object') {
    warnings.push('Field metadata should be an object');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
