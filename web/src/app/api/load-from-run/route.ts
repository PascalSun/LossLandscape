/**
 * API route to load landscape data from a run directory
 * Scans the run directory for .npz or .landscape files and loads them
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';

import { npzToJson } from '@/lib/npz';
import { 
  landscapeToJson, 
  ExportInProgressError, 
  ExportFailedError 
} from '@/lib/landscape-export';
import { importNpzData } from '@/lib/import-npz';
import fs from 'fs';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Find landscape files (.json, .landscape, or .npz) in a directory
 * Returns files in priority order: .json first, then .landscape, then .npz
 * Excludes metadata files like .export.meta.json
 */
function findLandscapeFiles(dir: string): { json?: string; landscape?: string; npz?: string } {
  const result: { json?: string; landscape?: string; npz?: string } = {};
  
  if (!existsSync(dir)) {
    return result;
  }

  try {
    const entries = readdirSync(dir);
    // Sort entries to prioritize actual data files over metadata files
    const sortedEntries = entries.sort((a, b) => {
      // Prioritize files that don't contain "meta" or "export.meta"
      const aIsMeta = a.includes('.export.meta.') || a.includes('.meta.');
      const bIsMeta = b.includes('.export.meta.') || b.includes('.meta.');
      if (aIsMeta && !bIsMeta) return 1;
      if (!aIsMeta && bIsMeta) return -1;
      return a.localeCompare(b);
    });
    
    for (const entry of sortedEntries) {
      const fullPath = join(dir, entry);
      try {
        const stat = statSync(fullPath);
        if (stat.isFile()) {
          // Skip metadata files
          if (entry.includes('.export.meta.') || entry.includes('.meta.')) {
            continue;
          }
          
          if (entry.endsWith('.json') && !result.json) {
            result.json = fullPath;
          } else if (entry.endsWith('.landscape') && !result.landscape) {
            result.landscape = fullPath;
          } else if (entry.endsWith('.npz') && !result.npz) {
            result.npz = fullPath;
          }
        }
      } catch (e) {
        console.warn(`Cannot access ${fullPath}:`, e);
      }
    }
  } catch (e) {
    console.error(`Error scanning directory ${dir}:`, e);
  }
  
  return result;
}

/**
 * Recursively find landscape files in a directory (depth-first), returning
 * the first match encountered (priority: json > landscape > npz).
 * Also returns candidate subdirectories that contain landscape files (for UI hints).
 */
function findLandscapeFilesRecursive(
  dir: string,
  maxDepth: number = 3
): { found?: { json?: string; landscape?: string; npz?: string }; candidates: string[] } {
  const candidates: string[] = [];

  const walk = (d: string, depth: number) => {
    if (depth > maxDepth) return;
    const local = findLandscapeFiles(d);
    if (local.json || local.landscape || local.npz) {
      candidates.push(d);
      // If we already found something earlier, don't override.
      return;
    }
    let entries: string[] = [];
    try {
      entries = readdirSync(d);
    } catch {
      return;
    }
    for (const entry of entries) {
      const fullPath = join(d, entry);
      try {
        const st = statSync(fullPath);
        if (st.isDirectory()) {
          walk(fullPath, depth + 1);
        }
      } catch {
        // ignore
      }
    }
  };

  walk(dir, 0);

  // If exactly one candidate dir has files, return that immediately.
  if (candidates.length === 1) {
    return { found: findLandscapeFiles(candidates[0]), candidates };
  }
  return { candidates };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const runPathInput = String(body?.runPath || '').trim();
    
    if (!runPathInput) {
      return NextResponse.json({ error: 'runPath is required' }, { status: 400 });
    }

    // Support repo-relative paths
    const runPath = path.isAbsolute(runPathInput)
      ? runPathInput
      : path.join(PROJECT_ROOT, runPathInput);

    if (!existsSync(runPath)) {
      return NextResponse.json({ error: `Run directory not found: ${runPathInput}` }, { status: 404 });
    }

    // Find landscape files in the run directory (priority: .json > .landscape > .npz)
    const landscapeFiles = findLandscapeFiles(runPath);
    
    if (!landscapeFiles.json && !landscapeFiles.landscape && !landscapeFiles.npz) {
      // The run directory may be a "container" that has multiple sub-runs (e.g. gravity_examples/*).
      const recursive = findLandscapeFilesRecursive(runPath, 4);
      if (recursive.found?.json || recursive.found?.landscape || recursive.found?.npz) {
        // Auto-load if there is exactly one subdir with files.
        const f = recursive.found;
        return NextResponse.json({
          success: true,
          id: undefined,
          runPath: runPathInput,
          dataFile: path.relative(PROJECT_ROOT, (f.json || f.landscape || f.npz) as string),
          data: JSON.parse(fs.readFileSync((f.json || f.landscape || f.npz) as string, 'utf-8')),
          note: 'Loaded from a nested subdirectory because the selected directory contained no landscape files.',
        });
      }

      return NextResponse.json({ 
        error: 'No landscape files (.json, .landscape, or .npz) found in the run directory',
        hint:
          recursive.candidates.length > 0
            ? {
                message:
                  'This directory contains subfolders that look like runs. Try selecting one of them instead.',
                candidates: recursive.candidates
                  .slice(0, 20)
                  .map((d) => path.relative(PROJECT_ROOT, d)),
              }
            : undefined,
      }, { status: 404 });
    }

    let data: any;
    let displayPath: string;
    let sourceFile: string;

    // Priority 1: Load .json file if available (fastest)
    if (landscapeFiles.json) {
      sourceFile = landscapeFiles.json;
      const relPath = path.relative(PROJECT_ROOT, sourceFile);
      displayPath = relPath.startsWith('..') ? path.basename(sourceFile) : relPath;
      
      const jsonContent = fs.readFileSync(sourceFile, 'utf-8');
      // Some legacy exports may contain NaN/Infinity; sanitize to keep JSON parseable
      const sanitizedContent = jsonContent
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');
      data = JSON.parse(sanitizedContent);
    }
    // Priority 2: Convert .landscape file to JSON
    else if (landscapeFiles.landscape) {
      sourceFile = landscapeFiles.landscape;
      const relPath = path.relative(PROJECT_ROOT, sourceFile);
      displayPath = relPath.startsWith('..') ? path.basename(sourceFile) : relPath;
      
      try {
        data = await landscapeToJson(sourceFile);
      } catch (e: any) {
        // Handle export-specific errors gracefully
        if (e instanceof ExportInProgressError) {
          return NextResponse.json(
            { 
              error: 'Export in progress',
              message: e.message,
              status: 'exporting',
              started_at: e.status.started_at,
              retry_after: 5000, // Suggest retrying after 5 seconds
            },
            { status: 202 } // 202 Accepted - request accepted but processing not complete
          );
        } else if (e instanceof ExportFailedError) {
          return NextResponse.json(
            { 
              error: 'Export failed',
              message: e.message,
              status: 'failed',
              error_details: e.status.error,
              failed_at: e.status.failed_at,
            },
            { status: 500 }
          );
        }
        // Re-throw other errors
        throw e;
      }
    }
    // Priority 3: Load .npz file
    else if (landscapeFiles.npz) {
      sourceFile = landscapeFiles.npz;
      const relPath = path.relative(PROJECT_ROOT, sourceFile);
      displayPath = relPath.startsWith('..') ? path.basename(sourceFile) : relPath;
      
      data = await npzToJson(sourceFile);
    } else {
      return NextResponse.json({ 
        error: 'No supported landscape files found' 
      }, { status: 404 });
    }

    // Normalize trajectory data format
    const normalizedData = {
      ...data,
      trajectory_1: data.trajectory_1 || data.trajectory_data?.traj_1,
      trajectory_2: data.trajectory_2 || data.trajectory_data?.traj_2,
      trajectory_3: data.trajectory_3 || data.trajectory_data?.traj_3,
      trajectory_epochs: data.trajectory_epochs || data.trajectory_data?.epochs,
    };

    // Validate data before proceeding
    // Check if data exists and has at least one valid grid (non-empty arrays)
    const has2D = Array.isArray(normalizedData.loss_grid_2d) && normalizedData.loss_grid_2d.length > 0;
    const has3D = Array.isArray(normalizedData.loss_grid_3d) && normalizedData.loss_grid_3d.length > 0;
    
    if (!normalizedData || (!has2D && !has3D)) {
      // Log for debugging
      console.warn('[load-from-run] Data validation failed:', {
        hasData: !!normalizedData,
        has2D,
        has3D,
        loss_grid_2d_type: typeof normalizedData?.loss_grid_2d,
        loss_grid_2d_length: Array.isArray(normalizedData?.loss_grid_2d) ? normalizedData.loss_grid_2d.length : 'not array',
        loss_grid_3d_type: typeof normalizedData?.loss_grid_3d,
        loss_grid_3d_length: Array.isArray(normalizedData?.loss_grid_3d) ? normalizedData.loss_grid_3d.length : 'not array',
        keys: normalizedData ? Object.keys(normalizedData) : 'no data',
      });
      
      return NextResponse.json(
        { 
          error: 'Invalid or incomplete data',
          message: 'The landscape file appears to be incomplete or corrupted. Please ensure the export process completed successfully.',
          debug: {
            has2D,
            has3D,
            loss_grid_2d_length: Array.isArray(normalizedData?.loss_grid_2d) ? normalizedData.loss_grid_2d.length : 'not array',
            loss_grid_3d_length: Array.isArray(normalizedData?.loss_grid_3d) ? normalizedData.loss_grid_3d.length : 'not array',
          }
        },
        { status: 422 } // 422 Unprocessable Entity
      );
    }

    // Import to database if needed (optional, for history tracking)
    const sourceLabel = `Run: ${runPathInput}`;
    
    let id: number | undefined;
    try {
      // Only import if data has required fields (for .npz files)
      if (normalizedData.X && normalizedData.Y && normalizedData.loss_grid_2d) {
        id = await importNpzData(normalizedData, sourceLabel);
      }
    } catch (e: any) {
      // Log but don't fail the request - database import is optional
      console.warn('Failed to save to database (continuing anyway):', e);
      // If it's a validation error, we might want to be more specific
      if (e.message?.includes('Cannot determine grid_size')) {
        console.warn('Data validation issue - grid_size cannot be determined from data structure');
      }
    }

    // Return the data for immediate display
    return NextResponse.json({
      success: true,
      id,
      runPath: runPathInput,
      dataFile: displayPath,
      data: normalizedData,
    });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

