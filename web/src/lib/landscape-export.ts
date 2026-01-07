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

const PYTHON_PATH = resolvePythonPath();

export async function landscapeToJson(landscapePath: string): Promise<any> {
  // Use Python to export .landscape file to JSON
  // The export command saves to a file (same name as .landscape but .json extension)
  const jsonPath = landscapePath.replace(/\.landscape$/, '.json');
  
  // Check if JSON file already exists (from previous export)
  if (fs.existsSync(jsonPath)) {
    const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
    return JSON.parse(jsonContent);
  }

  // Export .landscape to JSON using Python CLI
  await execFileAsync(
    PYTHON_PATH,
    ['-m', 'losslandscape', 'export', '--input', landscapePath, '--output', jsonPath],
    {
      cwd: PROJECT_ROOT,
      maxBuffer: 50 * 1024 * 1024,
    }
  );

  // Read the exported JSON file
  if (fs.existsSync(jsonPath)) {
    const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
    return JSON.parse(jsonContent);
  }

  throw new Error(`Failed to export landscape file: JSON file not created at ${jsonPath}`);
}

