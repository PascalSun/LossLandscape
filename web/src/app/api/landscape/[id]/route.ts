/**
 * API route to get loss landscape data by ID
 */

import { NextRequest, NextResponse } from 'next/server';
import { getLossLandscape, executeUpdate, executeDelete } from '@/lib/db';
import { readExportMetadata } from '@/lib/import-npz';
import path from 'path';
import { existsSync, readdirSync, statSync, readFileSync } from 'fs';
import { join } from 'path';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');

/**
 * Try to find and read the original JSON file to extract metadata and trajectory data
 */
async function tryLoadDataFromRunDir(runDir: string | undefined): Promise<{ metadata?: any; trajectory_losses?: number[]; trajectory_val_losses?: number[]; hessian?: any } | null> {
  if (!runDir) return null;
  
  try {
    // Resolve run directory path
    const runPath = path.isAbsolute(runDir)
      ? runDir
      : path.join(PROJECT_ROOT, runDir);
    
    if (!existsSync(runPath)) {
      return null;
    }
    
    // Find landscape files in the run directory (priority: .json)
    const landscapeFiles: { json?: string; } = {};
    
    try {
      const entries = readdirSync(runPath);
      for (const entry of entries) {
        const fullPath = join(runPath, entry);
        try {
          const stat = statSync(fullPath);
          if (stat.isFile()) {
            // Skip metadata files
            if (entry.includes('.export.meta.') || entry.includes('.meta.')) {
              continue;
            }
            
            if (entry.endsWith('.json') && !landscapeFiles.json) {
              landscapeFiles.json = fullPath;
            }
          }
        } catch (e) {
          // Ignore individual file errors
        }
      }
    } catch (e) {
      // Ignore directory read errors
    }
    
    // Try to read from JSON file first (most likely to have metadata and trajectory data)
    let jsonFilePath: string | undefined;
    if (landscapeFiles.json) {
      jsonFilePath = landscapeFiles.json;
    }
    
    if (jsonFilePath && existsSync(jsonFilePath)) {
      try {
        // Read the JSON file
        const jsonContent = readFileSync(jsonFilePath, 'utf-8');
        // Sanitize NaN/Infinity for JSON parsing
        const sanitizedContent = jsonContent
          .replace(/\bNaN\b/g, 'null')
          .replace(/\bInfinity\b/g, 'null')
          .replace(/\b-Infinity\b/g, 'null');
        const jsonData = JSON.parse(sanitizedContent);
        
        // Extract metadata and trajectory data
        const result: { metadata?: any; trajectory_losses?: number[]; trajectory_val_losses?: number[]; hessian?: any } = {};
        
        if (jsonData.metadata) {
          result.metadata = jsonData.metadata;
        }
        
        if (jsonData.trajectory_data) {
          if (jsonData.trajectory_data.losses) {
            result.trajectory_losses = jsonData.trajectory_data.losses;
          }
          if (jsonData.trajectory_data.val_losses) {
            result.trajectory_val_losses = jsonData.trajectory_data.val_losses;
          }
        }
        
        // Also check top-level trajectory_losses and trajectory_val_losses
        if (jsonData.trajectory_losses) {
          result.trajectory_losses = jsonData.trajectory_losses;
        }
        if (jsonData.trajectory_val_losses) {
          result.trajectory_val_losses = jsonData.trajectory_val_losses;
        }

        if (jsonData.hessian) {
          result.hessian = jsonData.hessian;
        }
        
        return Object.keys(result).length > 0 ? result : null;
      } catch (e) {
        console.warn(`Failed to parse JSON file ${jsonFilePath}:`, e);
      }
    }
  } catch (e) {
    console.warn(`Failed to load data from run_dir ${runDir}:`, e);
  }
  
  return null;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: idParam } = await params;
    const id = parseInt(idParam);
    
    if (isNaN(id)) {
      return NextResponse.json(
        { error: 'Invalid ID' },
        { status: 400 }
      );
    }

    let data = await getLossLandscape(id);

    if (!data) {
      return NextResponse.json(
        { error: 'Landscape not found' },
        { status: 404 }
      );
    }

    // Try to load metadata and trajectory data from original files if missing
    // Note: This is optional - if files don't exist (e.g., deleted after import),
    // the route will still return data from the database (standalone mode)
    let loadedMetadata: any = undefined;
    let loadedHessian: any = undefined;
    let loadedConfigYml: string | undefined = undefined;
    if (data.run_dir) {
      const loadedData = await tryLoadDataFromRunDir(data.run_dir);
      
      if (loadedData) {
        // Store metadata for response
        if (loadedData.metadata) {
          loadedMetadata = loadedData.metadata;
          
          // Store in database for future use (async, don't wait)
          if (!data.export_metadata) {
            executeUpdate(
              `UPDATE loss_landscape_data SET export_metadata = ? WHERE id = ?`,
              JSON.stringify(loadedData.metadata),
              id
            ).catch((err) => {
              console.warn(`Failed to update metadata in database for id ${id}:`, err);
            });
          }
        }

        if (loadedData.hessian && !data.hessian) {
          loadedHessian = loadedData.hessian;
          // Persist hessian so future reads don't rely on filesystem.
          executeUpdate(
            `UPDATE loss_landscape_data SET hessian = ? WHERE id = ?`,
            JSON.stringify(loadedData.hessian),
            id
          ).catch((err) => {
            console.warn(`Failed to update hessian in database for id ${id}:`, err);
          });
        }
        
        // Add trajectory losses if available
        if (loadedData.trajectory_losses && data.trajectory_data) {
          data.trajectory_data = {
            ...data.trajectory_data,
            losses: loadedData.trajectory_losses,
          };
        }
        if (loadedData.trajectory_val_losses && data.trajectory_data) {
          data.trajectory_data = {
            ...data.trajectory_data,
            val_losses: loadedData.trajectory_val_losses,
          };
        }
      }
      
      // Try to load config.yml/config.yaml from run directory if not in database
      if (!data.config_yml) {
        try {
          const runPath = path.isAbsolute(data.run_dir)
            ? data.run_dir
            : path.join(PROJECT_ROOT, data.run_dir);
          const configYmlPath = path.join(runPath, 'config.yml');
          const configYamlPath = path.join(runPath, 'config.yaml');
          
          if (existsSync(configYmlPath)) {
            loadedConfigYml = readFileSync(configYmlPath, 'utf-8');
          } else if (existsSync(configYamlPath)) {
            loadedConfigYml = readFileSync(configYamlPath, 'utf-8');
          }
          
          // Store in database for future use (async, don't wait)
          if (loadedConfigYml) {
            executeUpdate(
              `UPDATE loss_landscape_data SET config_yml = ? WHERE id = ?`,
              loadedConfigYml,
              id
            ).catch((err) => {
              console.warn(`Failed to update config_yml in database for id ${id}:`, err);
            });
          }
        } catch (e) {
          console.warn(`Failed to read config.yml from run_dir ${data.run_dir}:`, e);
        }
      }
    }

    // Transform the data to match frontend expectations
    // Frontend expects 'metadata' (from JSON file), not 'export_metadata' (from .export.meta.json)
    const responseData: any = {
      ...data,
      hessian: data.hessian || loadedHessian,
      config_yml: data.config_yml || loadedConfigYml,
      // Use loaded metadata if available, otherwise try to extract from export_metadata
      metadata: loadedMetadata || data.export_metadata?.metadata || 
        (data.export_metadata && typeof data.export_metadata === 'object' && 
         !data.export_metadata.status && !data.export_metadata.started_at ? data.export_metadata : undefined),
    };

    return NextResponse.json(responseData);
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: idParam } = await params;
    const id = parseInt(idParam);
    
    if (isNaN(id)) {
      return NextResponse.json(
        { error: 'Invalid ID' },
        { status: 400 }
      );
    }

    // Delete trajectory_points first (foreign key constraint)
    await executeDelete(
      `DELETE FROM trajectory_points WHERE landscape_id = ?`,
      id
    ).catch((err) => {
      console.warn(`Failed to delete trajectory_points for id ${id}:`, err);
      // Continue even if this fails
    });

    // Delete the landscape record
    await executeDelete(
      `DELETE FROM loss_landscape_data WHERE id = ?`,
      id
    );

    return NextResponse.json({ success: true, id });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
