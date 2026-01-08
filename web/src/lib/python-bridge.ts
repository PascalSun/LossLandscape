/**
 * Bridge to call Python loss landscape generation
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

function resolvePythonPath(): string {
  if (process.env.PYTHON_PATH && process.env.PYTHON_PATH.trim()) {
    return process.env.PYTHON_PATH.trim();
  }
  // Prefer the repo venv if present: <repo>/.venv/bin/python
  const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  return 'python';
}

const PYTHON_PATH = resolvePythonPath();

export interface GenerateLandscapeOptions {
  config?: string;
  run?: string;
  checkpoint?: string;
  mode?: '2d' | '3d';
  gridSize?: number;
  direction?: string;
  rangeScale?: number;
  maxBatches?: number;
}

export interface LandscapeResult {
  success: boolean;
  outputDir?: string;
  dataFile?: string;
  error?: string;
}

export async function generateLossLandscape(
  options: GenerateLandscapeOptions
): Promise<LandscapeResult> {
  try {
    // Build command
    const cmdParts = [
      PYTHON_PATH,
      '-m',
      'pinn_bdke',
      'utils',
      'loss-landscape',
    ];

    if (options.config) {
      cmdParts.push('--config', options.config);
    } else if (options.run) {
      cmdParts.push('--run', options.run);
    } else {
      throw new Error('Either config or run must be provided');
    }

    if (options.checkpoint) {
      cmdParts.push('--checkpoint', options.checkpoint);
    }

    if (options.mode) {
      cmdParts.push('--mode', options.mode);
    }

    if (options.gridSize) {
      cmdParts.push('--grid-size', options.gridSize.toString());
    }

    if (options.direction) {
      cmdParts.push('--direction', options.direction);
    }

    if (options.rangeScale) {
      cmdParts.push('--range-scale', options.rangeScale.toString());
    }

    if (options.maxBatches) {
      cmdParts.push('--max-batches', options.maxBatches.toString());
    }

    const command = cmdParts.join(' ');
    
    console.log(`Executing: ${command}`);
    console.log(`Working directory: ${PROJECT_ROOT}`);

    const { stdout, stderr } = await execAsync(command, {
      cwd: PROJECT_ROOT,
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
    });

    console.log('Python output:', stdout);
    if (stderr) {
      console.error('Python stderr:', stderr);
    }

    // Try to extract output directory from stdout
    const outputMatch = stdout.match(/Output directory: (.+)/);
    const outputDir = outputMatch ? outputMatch[1] : undefined;

    // Look for data file
    const dataFileMatch = stdout.match(/Saved data to: (.+)/);
    const dataFile = dataFileMatch ? dataFileMatch[1] : undefined;

    if (outputDir && dataFile && fs.existsSync(dataFile)) {
      return {
        success: true,
        outputDir,
        dataFile,
      };
    }

    return {
      success: true,
      outputDir,
      dataFile,
    };
  } catch (error: any) {
    console.error('Error generating loss landscape:', error);
    return {
      success: false,
      error: error.message || String(error),
    };
  }
}


