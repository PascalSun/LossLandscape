/**
 * API route to read config file content
 */

import { NextRequest, NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const configPath = searchParams.get('configPath') || searchParams.get('path'); // Support both for backward compatibility
    const runDir = searchParams.get('runDir');

    if (!configPath && !runDir) {
      return NextResponse.json(
        { error: 'Either configPath or runDir is required' },
        { status: 400 }
      );
    }

    const repoRoot = process.env.REPO_ROOT || process.cwd().replace(/\/app$/, '');
    let filePath: string;

    if (runDir) {
      // If runDir is provided, look for config.yml in the run directory
      const runPath = join(repoRoot, runDir);
      const configInRun = join(runPath, 'config.yml');
      if (existsSync(configInRun)) {
        filePath = configInRun;
      } else {
        // Try config.yaml
        const configYaml = join(runPath, 'config.yaml');
        if (existsSync(configYaml)) {
          filePath = configYaml;
        } else {
          return NextResponse.json(
            { error: `Config file not found in run directory: ${runDir}` },
            { status: 404 }
          );
        }
      }
    } else if (configPath) {
      // Use provided config path
      filePath = configPath.startsWith('/') || configPath.startsWith('\\')
        ? configPath
        : join(repoRoot, configPath);
    } else {
      return NextResponse.json(
        { error: 'Either configPath or runDir is required' },
        { status: 400 }
      );
    }

    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: `Config file not found: ${filePath}` },
        { status: 404 }
      );
    }

    const content = readFileSync(filePath, 'utf-8');
    const relativePath = filePath.replace(repoRoot + '/', '').replace(repoRoot + '\\', '');

    return NextResponse.json({
      success: true,
      content,
      path: relativePath,
      absolutePath: filePath,
    });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

