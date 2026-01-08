/**
 * Convert .landscape (DuckDB) files to JSON by calling Python.
 */

import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execFileAsync = promisify(execFile);

const PROJECT_ROOT =
  process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

function resolvePythonPath(): string {
  if (process.env.PYTHON_PATH && process.env.PYTHON_PATH.trim()) {
    return process.env.PYTHON_PATH.trim();
  }
  const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  return 'python';
}

function resolveLossvisCommand(): string {
  // Try to find lossvis command in venv first
  const venvLossvis = path.join(PROJECT_ROOT, '.venv', 'bin', 'lossvis');
  if (fs.existsSync(venvLossvis)) return venvLossvis;
  
  // Fallback to system lossvis
  return 'lossvis';
}

const PYTHON_PATH = resolvePythonPath();
const LOSSVIS_CMD = resolveLossvisCommand();

interface ExportStatus {
  status: 'exporting' | 'completed' | 'failed';
  started_at?: string;
  completed_at?: string;
  failed_at?: string;
  export_duration_seconds?: number;
  error?: string;
  metadata?: {
    grid_size_2d?: number;
    grid_size_3d?: number;
    trajectory_points?: number;
    baseline_loss?: number;
    mode?: string;
  };
}

function getStatusFilePath(jsonPath: string): string {
  return jsonPath.replace(/\.json$/, '.export.meta.json');
}

function checkExportStatus(jsonPath: string): ExportStatus | null {
  const statusPath = getStatusFilePath(jsonPath);
  if (!fs.existsSync(statusPath)) {
    return null;
  }
  
  try {
    const statusContent = fs.readFileSync(statusPath, 'utf-8');
    return JSON.parse(statusContent) as ExportStatus;
  } catch (e) {
    // If status file is corrupted, ignore it
    return null;
  }
}

export class ExportInProgressError extends Error {
  constructor(
    public readonly status: ExportStatus,
    message?: string
  ) {
    super(message || 'Export is currently in progress. Please wait for it to complete.');
    this.name = 'ExportInProgressError';
  }
}

export class ExportFailedError extends Error {
  constructor(
    public readonly status: ExportStatus,
    message?: string
  ) {
    super(message || `Export failed: ${status.error || 'Unknown error'}`);
    this.name = 'ExportFailedError';
  }
}

export async function landscapeToJson(landscapePath: string): Promise<any> {
  // Use Python to export .landscape file to JSON
  // The export command saves to a file (same name as .landscape but .json extension)
  const jsonPath = landscapePath.replace(/\.landscape$/, '.json');
  const statusPath = getStatusFilePath(jsonPath);
  
  // Check export status first
  const existingStatus = checkExportStatus(jsonPath);
  if (existingStatus) {
    if (existingStatus.status === 'exporting') {
      // Check if export process is still running (status file was recently updated)
      const statusStat = fs.statSync(statusPath);
      const now = Date.now();
      const statusAge = now - statusStat.mtimeMs;
      
      // If status file is older than 5 minutes, assume export is stuck
      if (statusAge > 5 * 60 * 1000) {
        // Remove stale status file and retry
        try {
          fs.unlinkSync(statusPath);
        } catch (e) {
          // Ignore errors
        }
      } else {
        throw new ExportInProgressError(
          existingStatus,
          `Export is in progress (started at ${existingStatus.started_at}). Please wait...`
        );
      }
    } else if (existingStatus.status === 'failed') {
      // If previous export failed, we can retry
      // But inform the caller about the previous failure
      throw new ExportFailedError(
        existingStatus,
        `Previous export failed: ${existingStatus.error || 'Unknown error'}`
      );
    }
  }
  
  // Check if JSON file already exists and is complete
  if (fs.existsSync(jsonPath)) {
    const status = checkExportStatus(jsonPath);
    // If status says completed, or no status file (legacy export), use the file
    if (!status || status.status === 'completed') {
      try {
        const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
        const data = JSON.parse(jsonContent);
        
        // Validate that the JSON is complete (has required fields)
        if (data.grid_size || data.loss_grid_2d || data.loss_grid_3d) {
          return data;
        }
        // If JSON exists but is incomplete, continue to re-export
      } catch (e) {
        // If JSON is corrupted, continue to re-export
      }
    }
  }

  // Export .landscape to JSON using CLI command
  // Use lossvis command directly (it's installed as a script entry point)
  try {
    await execFileAsync(
      LOSSVIS_CMD,
      ['export', '--input', landscapePath, '--output', jsonPath],
      {
        cwd: PROJECT_ROOT,
        maxBuffer: 50 * 1024 * 1024,
      }
    );
  } catch (e: any) {
    // Check if export failed and status file has error info
    const status = checkExportStatus(jsonPath);
    if (status?.status === 'failed') {
      throw new ExportFailedError(status);
    }
    throw new Error(`Export command failed: ${e.message || String(e)}`);
  }

  // Read the exported JSON file
  if (fs.existsSync(jsonPath)) {
    const status = checkExportStatus(jsonPath);
    if (status?.status === 'completed') {
      const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
      return JSON.parse(jsonContent);
    } else if (status?.status === 'failed') {
      throw new ExportFailedError(status);
    } else {
      // Status file might not be updated yet, but JSON exists - try to read it
      try {
        const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
        return JSON.parse(jsonContent);
      } catch (e) {
        throw new Error(`Exported JSON file is invalid or incomplete`);
      }
    }
  }

  throw new Error(`Failed to export landscape file: JSON file not created at ${jsonPath}`);
}

