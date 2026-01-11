/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * API route to generate loss landscape
 */

import { NextRequest, NextResponse } from 'next/server';
import { generateLossLandscape, GenerateLandscapeOptions } from '@/lib/python-bridge';
import { saveLossLandscape, getLossLandscape } from '@/lib/db';
import { existsSync, readFileSync } from 'fs';
import { unlink } from 'fs/promises';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const options: GenerateLandscapeOptions = {
      config: body.config,
      run: body.run,
      checkpoint: body.checkpoint,
      mode: body.mode || '2d',
      gridSize: body.gridSize || 50,
      direction: body.direction || 'auto',
      rangeScale: body.rangeScale || 1.0,
      maxBatches: body.maxBatches,
    };

    // Generate loss landscape
    const result = await generateLossLandscape(options);

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to generate loss landscape' },
        { status: 500 }
      );
    }

    // Load the generated data
    if (result.dataFile) {
      // Load JSON file
      const content = readFileSync(result.dataFile, 'utf-8');
      // Sanitize NaN/Infinity
      const sanitized = content
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');
      const data = JSON.parse(sanitized);
      
      // Transform trajectory data from npz format to expected format
      // Note: If Python now generates standard JSON format, this might be redundant or require adjustment
      // Checks for both snake_case and camelCase or flattened properties
      const trajectory_data = data.trajectory_data || ((data.trajectory_1 && data.trajectory_2) ? {
        traj_1: data.trajectory_1,
        traj_2: data.trajectory_2,
        traj_3: data.trajectory_3,
        epochs: data.trajectory_epochs || [],
      } : undefined);
      
      // Save to database
      // Ensure run_dir is a string (it might be a Path object or undefined)
      const runDirStr = options.run ? String(options.run) : undefined;
      const repoRoot =
        process.env.REPO_ROOT || process.cwd().replace(/\/app$/, '');

      // If the request is "from run", also persist the resolved config path from that run
      // so history can show "which config was used".
      const configPathStr = (() => {
        const direct = options.config ? String(options.config) : undefined;
        if (direct) return direct;
        if (!runDirStr) return undefined;

        const runAbs = path.isAbsolute(runDirStr)
          ? runDirStr
          : path.join(repoRoot, runDirStr);

        const candidates = [
          'config_resolved.yaml',
          'config_resolved.yml',
          'config.yaml',
          'config.yml',
        ].map((name) => path.join(runAbs, name));

        const foundAbs = candidates.find((p) => existsSync(p));
        if (!foundAbs) return undefined;

        // Store relative path when possible (nicer display, portable)
        const rel = path.relative(repoRoot, foundAbs);
        return rel.startsWith('..') ? foundAbs : rel;
      })();
      
      console.log('[generate/route] Saving to database:', {
        config_path: configPathStr,
        run_dir: runDirStr,
        mode: options.mode || '2d',
        hasTrajectory: !!trajectory_data,
      });
      
      let id: number;
      try {
        id = await saveLossLandscape({
          config_path: configPathStr,
          run_dir: runDirStr,
          mode: options.mode || '2d',
          direction: options.direction || 'auto',
          grid_size: options.gridSize || 50,
          X: data.X,
          Y: data.Y,
          Z: data.Z,
          loss_grid_2d: data.loss_grid_2d,
          loss_grid_3d: data.loss_grid_3d,
          baseline_loss: data.baseline_loss,
          loss_line_1d: data.loss_line_1d,
          X_1d: data.X_1d,
          baseline_loss_1d: data.baseline_loss_1d,
          trajectory_data,
          import_source: 'generate',
          imported_at: new Date(),
          skipDuplicateCheck: true, // Always create new record for generate API
        });
        console.log('[generate/route] Saved to database with id:', id);
        
        // Verify data is actually in DuckDB before deleting
        const savedData = await getLossLandscape(id);
        if (!savedData) {
          console.warn(`[generate/route] Failed to verify data was saved to database (id=${id}), skipping file deletion`);
        } else if (result.dataFile) {
          // Delete the generated JSON file after successful save to database and verification
          try {
            await unlink(result.dataFile);
            console.log(`[generate/route] Deleted generated file after successful save to DuckDB: ${result.dataFile}`);
          } catch (deleteError: any) {
            console.warn(`[generate/route] Failed to delete generated file ${result.dataFile}:`, deleteError);
            // Don't fail the request if deletion fails
          }
        }
      } catch (saveError: any) {
        console.error('[generate/route] Failed to save to database:', saveError);
        // Continue even if save fails, but log the error
        throw new Error(`Failed to save to database: ${saveError.message}`);
      }

      // Return normalized data for frontend
      const normalizedData = {
        ...data,
        trajectory_1: data.trajectory_1,
        trajectory_2: data.trajectory_2,
        trajectory_3: data.trajectory_3,
        trajectory_epochs: data.trajectory_epochs,
      };

      return NextResponse.json({
        success: true,
        id,
        outputDir: result.outputDir,
        dataFile: result.dataFile,
        data: normalizedData,
      });
    }

    return NextResponse.json({
      success: true,
      outputDir: result.outputDir,
    });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
