/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * API route to scan a folder and automatically import all JSON files
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';
import { unlink } from 'fs/promises';

import { importNpzData } from '@/lib/import-npz';
import { getLossLandscape } from '@/lib/db';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Recursively find all .json files in a directory
 */
function findLandscapeFiles(dir: string, baseDir: string = dir): string[] {
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
          files.push(...findLandscapeFiles(fullPath, baseDir));
        } else if (entry.endsWith('.json') && !entry.includes('.export.meta.') && !entry.includes('.meta.')) {
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

    // Find all .json files
    const jsonFiles = findLandscapeFiles(folderPath);
    
    if (jsonFiles.length === 0) {
      return NextResponse.json({ 
        success: true, 
        files: [], 
        imported: 0,
        message: 'No .json files found in the specified folder' 
      });
    }

    // Import each file
    const imported: number[] = [];
    const errors: string[] = [];

    for (const jsonFile of jsonFiles) {
      try {
        const content = await import('fs/promises').then(fs => fs.readFile(jsonFile, 'utf-8'));
        // Sanitize NaN/Infinity
        const sanitized = content
          .replace(/\bNaN\b/g, 'null')
          .replace(/\bInfinity\b/g, 'null')
          .replace(/\b-Infinity\b/g, 'null');
        const data = JSON.parse(sanitized);
        
        // Get relative path for display
        const relPath = path.relative(PROJECT_ROOT, jsonFile);
        const displayPath = relPath.startsWith('..') ? path.basename(jsonFile) : relPath;

        // Import using unified import function
        const sourceLabel = `Scanned: ${displayPath}`;
        const filename = path.basename(jsonFile);
        const id = await importNpzData(data, sourceLabel, jsonFile, undefined, undefined, 'scan', filename);

        imported.push(id);

        // Verify data is actually in DuckDB before deleting
        const savedData = await getLossLandscape(id);
        if (!savedData) {
          console.warn(`Failed to verify data was saved to database (id=${id}), skipping file deletion for ${jsonFile}`);
        } else {
          // Delete the original JSON file after successful import and verification
          try {
            await unlink(jsonFile);
            console.log(`Deleted original file after successful import to DuckDB: ${jsonFile}`);
          } catch (deleteError: any) {
            console.warn(`Failed to delete file ${jsonFile} after import:`, deleteError);
            // Don't fail the request if deletion fails
          }
        }
      } catch (error: any) {
        console.error(`Failed to import ${jsonFile}:`, error);
        errors.push(`${path.basename(jsonFile)}: ${error.message}`);
      }
    }

    return NextResponse.json({
      success: true,
      files: jsonFiles.map(f => path.relative(PROJECT_ROOT, f)),
      imported: imported.length,
      total: jsonFiles.length,
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

