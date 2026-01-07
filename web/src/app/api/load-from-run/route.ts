/**
 * API route to load landscape data from a run directory
 * Scans the run directory for .npz or .landscape files and loads them
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';

import { npzToJson } from '@/lib/npz';
import { landscapeToJson } from '@/lib/landscape-export';
import { importNpzData } from '@/lib/import-npz';
import fs from 'fs';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Find landscape files (.json, .landscape, or .npz) in a directory
 * Returns files in priority order: .json first, then .landscape, then .npz
 */
function findLandscapeFiles(dir: string): { json?: string; landscape?: string; npz?: string } {
  const result: { json?: string; landscape?: string; npz?: string } = {};
  
  if (!existsSync(dir)) {
    return result;
  }

  try {
    const entries = readdirSync(dir);
    for (const entry of entries) {
      const fullPath = join(dir, entry);
      try {
        const stat = statSync(fullPath);
        if (stat.isFile()) {
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
      return NextResponse.json({ 
        error: 'No landscape files (.json, .landscape, or .npz) found in the run directory' 
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
      data = JSON.parse(jsonContent);
    }
    // Priority 2: Convert .landscape file to JSON
    else if (landscapeFiles.landscape) {
      sourceFile = landscapeFiles.landscape;
      const relPath = path.relative(PROJECT_ROOT, sourceFile);
      displayPath = relPath.startsWith('..') ? path.basename(sourceFile) : relPath;
      
      data = await landscapeToJson(sourceFile);
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

    // Import to database if needed (optional, for history tracking)
    const sourceLabel = `Run: ${runPathInput}`;
    
    let id: number | undefined;
    try {
      // Only import if data has required fields (for .npz files)
      if (normalizedData.X && normalizedData.Y && normalizedData.loss_grid_2d) {
        id = await importNpzData(normalizedData, sourceLabel);
      }
    } catch (e) {
      console.warn('Failed to save to database (continuing anyway):', e);
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

