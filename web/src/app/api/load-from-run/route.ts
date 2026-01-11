/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * API route to load landscape data from a run directory
 * Scans the run directory for .json files and loads them
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';

import { importNpzData } from '@/lib/import-npz';
import fs from 'fs';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Find landscape files (.json) in a directory
 * Returns files: .json
 * Excludes metadata files like .export.meta.json
 */
function findLandscapeFiles(dir: string): { json?: string; } {
  const result: { json?: string; } = {};
  
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
 * the first match encountered (priority: json).
 * Also returns candidate subdirectories that contain landscape files (for UI hints).
 */
function findLandscapeFilesRecursive(
  dir: string,
  maxDepth: number = 3
): { found?: { json?: string; }; candidates: string[] } {
  const candidates: string[] = [];

  const walk = (d: string, depth: number) => {
    if (depth > maxDepth) return;
    const local = findLandscapeFiles(d);
    if (local.json) {
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

    // Find landscape files in the run directory (priority: .json)
    const landscapeFiles = findLandscapeFiles(runPath);
    
    if (!landscapeFiles.json) {
      // The run directory may be a "container" that has multiple sub-runs (e.g. gravity_examples/*).
      const recursive = findLandscapeFilesRecursive(runPath, 4);
      if (recursive.found?.json) {
        // Auto-load if there is exactly one subdir with files.
        const f = recursive.found;
        return NextResponse.json({
          success: true,
          id: undefined,
          runPath: runPathInput,
          dataFile: path.relative(PROJECT_ROOT, (f.json) as string),
          data: JSON.parse(fs.readFileSync((f.json) as string, 'utf-8')),
          note: 'Loaded from a nested subdirectory because the selected directory contained no landscape files.',
        });
      }

      return NextResponse.json({ 
        error: 'No landscape files (.json) found in the run directory',
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
    } else {
      return NextResponse.json({ 
        error: 'No supported landscape files found' 
      }, { status: 404 });
    }

    // Read config.yml/config.yaml from run directory if available
    let configYml: string | undefined = undefined;
    try {
      const configYmlPath = path.join(runPath, 'config.yml');
      const configYamlPath = path.join(runPath, 'config.yaml');
      
      if (fs.existsSync(configYmlPath)) {
        configYml = fs.readFileSync(configYmlPath, 'utf-8');
      } else if (fs.existsSync(configYamlPath)) {
        configYml = fs.readFileSync(configYamlPath, 'utf-8');
      }
    } catch (e) {
      // Silently fail if config.yml cannot be read
      console.warn('[load-from-run] Failed to read config.yml:', e);
    }

    // Normalize trajectory data format
    const normalizedData = {
      ...data,
      trajectory_1: data.trajectory_1 || data.trajectory_data?.traj_1,
      trajectory_2: data.trajectory_2 || data.trajectory_data?.traj_2,
      trajectory_3: data.trajectory_3 || data.trajectory_data?.traj_3,
      trajectory_epochs: data.trajectory_epochs || data.trajectory_data?.epochs,
      trajectory_losses: data.trajectory_losses || data.trajectory_data?.losses,
      trajectory_val_losses: data.trajectory_val_losses || data.trajectory_data?.val_losses,
      // Preserve trajectory_data structure if it exists
      trajectory_data: data.trajectory_data,
      // Preserve metadata if it exists
      metadata: data.metadata,
      // Include config_yml if found
      config_yml: configYml,
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
      // Only import if data has required fields
      if (normalizedData.X && normalizedData.Y && normalizedData.loss_grid_2d) {
        const filename = path.basename(sourceFile);
        id = await importNpzData(normalizedData, sourceLabel, sourceFile, undefined, runPathInput, 'run', filename);
        
        // Note: We do NOT delete run files after import - they should be preserved
        // as they are part of the run directory structure and may be needed for other purposes
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

