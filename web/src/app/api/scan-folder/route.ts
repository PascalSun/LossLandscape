/**
 * API route to scan a folder and automatically import all .npz files
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';

import { npzToJson } from '@/lib/npz';
import { importNpzData } from '@/lib/import-npz';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Recursively find all .npz files in a directory
 */
function findNpzFiles(dir: string, baseDir: string = dir): string[] {
  const files: string[] = [];
  
  if (!existsSync(dir)) {
    return files;
  }

  try {
    const entries = readdirSync(dir);
    for (const entry of entries) {
      const fullPath = join(dir, entry);
      try {
        const stat = statSync(fullPath);
        if (stat.isDirectory()) {
          // Recursively scan subdirectories
          files.push(...findNpzFiles(fullPath, baseDir));
        } else if (entry.endsWith('.npz')) {
          files.push(fullPath);
        }
      } catch (e) {
        // Skip files/directories we can't access
        console.warn(`Cannot access ${fullPath}:`, e);
      }
    }
  } catch (e) {
    console.error(`Error scanning directory ${dir}:`, e);
  }
  
  return files;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const folderPathInput = String(body?.folderPath || '').trim();
    
    if (!folderPathInput) {
      return NextResponse.json({ error: 'folderPath is required' }, { status: 400 });
    }

    // Support repo-relative paths
    const folderPath = path.isAbsolute(folderPathInput)
      ? folderPathInput
      : path.join(PROJECT_ROOT, folderPathInput);

    if (!existsSync(folderPath)) {
      return NextResponse.json({ error: `Folder not found: ${folderPathInput}` }, { status: 404 });
    }

    // Find all .npz files
    const npzFiles = findNpzFiles(folderPath);
    
    if (npzFiles.length === 0) {
      return NextResponse.json({ 
        success: true, 
        files: [], 
        imported: 0,
        message: 'No .npz files found in the specified folder' 
      });
    }

    // Import each file
    const imported: number[] = [];
    const errors: string[] = [];

    for (const npzFile of npzFiles) {
      try {
        const data = await npzToJson(npzFile);
        
        // Get relative path for display
        const relPath = path.relative(PROJECT_ROOT, npzFile);
        const displayPath = relPath.startsWith('..') ? path.basename(npzFile) : relPath;

        // Import using unified import function
        const sourceLabel = `Scanned: ${displayPath}`;
        const id = await importNpzData(data, sourceLabel);

        imported.push(id);
      } catch (error: any) {
        console.error(`Failed to import ${npzFile}:`, error);
        errors.push(`${path.basename(npzFile)}: ${error.message}`);
      }
    }

    return NextResponse.json({
      success: true,
      files: npzFiles.map(f => path.relative(PROJECT_ROOT, f)),
      imported: imported.length,
      total: npzFiles.length,
      errors: errors.length > 0 ? errors : undefined,
    });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

