/* eslint-disable @typescript-eslint/no-explicit-any */
import { NextResponse } from 'next/server';
import { readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';
import path from 'path';

export const dynamic = 'force-dynamic';

/**
 * GET /api/list-runs
 * Returns: { configs: string[], runs: string[] }
 */
export async function GET() {
  try {
    // Resolve repo root:
    // - If REPO_ROOT is provided, use it.
    // - Otherwise, if cwd ends with "/web", strip it to move to repo root.
    // - Else, fallback to cwd.
    const repoRoot = (() => {
      if (process.env.REPO_ROOT) return process.env.REPO_ROOT;
      const cwd = process.cwd();
      if (cwd.endsWith('/web')) return path.join(cwd, '..');
      return cwd;
    })();
    
    // Scan configs directory
    const configsDir = join(repoRoot, 'configs');
    let configs: string[] = [];
    if (existsSync(configsDir)) {
      const scan = (dir: string, prefix = ''): void => {
        const entries = readdirSync(dir);
        for (const entry of entries) {
          const fullPath = join(dir, entry);
          const stat = statSync(fullPath);
          if (stat.isDirectory()) {
            scan(fullPath, prefix ? `${prefix}/${entry}` : entry);
          } else if (entry.endsWith('.yml') || entry.endsWith('.yaml')) {
            configs.push(prefix ? `configs/${prefix}/${entry}` : `configs/${entry}`);
          }
        }
      };
      scan(configsDir);
    }

    // Scan outputs directory for run folders
    const outputsDir = join(repoRoot, 'outputs');
    let runs: string[] = [];
    let runsTree: any = {};
    
    if (existsSync(outputsDir)) {
      const hasLandscapeFiles = (dir: string): boolean => {
        try {
          const entries = readdirSync(dir);
          for (const entry of entries) {
            // Skip metadata files
            if (entry.includes('.export.meta.') || entry.includes('.meta.')) continue;
            if (entry.endsWith('.json')) {
              return true;
            }
          }
        } catch {
          // ignore
        }
        return false;
      };

      // Build flat list for backward compatibility
      const scanOutputs = (dir: string, prefix = 'outputs'): void => {
        const entries = readdirSync(dir);
        for (const entry of entries) {
          const fullPath = join(dir, entry);
          const stat = statSync(fullPath);
          if (stat.isDirectory()) {
            // Prefer listing directories that actually contain landscape files.
            // This supports nested layouts like outputs/run_xxx/foo/complete_example.json.
            if (hasLandscapeFiles(fullPath)) {
              runs.push(`${prefix}/${entry}`);
            }
            // Recursively scan subdirectories
            scanOutputs(fullPath, `${prefix}/${entry}`);
          }
        }
      };
      scanOutputs(outputsDir);
      
      // Sort runs by name (most recent first if using timestamp format)
      runs = Array.from(new Set(runs)).sort().reverse();

      // Build tree structure for grouped display
      const buildTree = (dir: string, prefix = 'outputs'): any => {
        const tree: any = { children: {}, paths: [] };
        try {
          const entries = readdirSync(dir).sort();
          for (const entry of entries) {
            const fullPath = join(dir, entry);
            const stat = statSync(fullPath);
            if (stat.isDirectory()) {
              const currentPath = `${prefix}/${entry}`;
              const hasFiles = hasLandscapeFiles(fullPath);
              
              // Recursively build subtree first
              const subtree = buildTree(fullPath, currentPath);
              const hasSubItems = subtree.paths.length > 0 || Object.keys(subtree.children).length > 0;
              
              // If this directory has files directly, add to paths
              if (hasFiles) {
                tree.paths.push(currentPath);
              }
              
              // If there are sub-items (files in subdirs or nested folders), add as child
              if (hasSubItems) {
                tree.children[entry] = subtree;
              } else if (hasFiles) {
                // If only has files but no sub-items, still add as child for consistency
                tree.children[entry] = { children: {}, paths: [] };
              }
            }
          }
        } catch {
          // ignore
        }
        return tree;
      };
      
      runsTree = buildTree(outputsDir);
    }

    return NextResponse.json({ configs, runs, runsTree });
  } catch (error: any) {
    console.error('Error listing runs/configs:', error);
    return NextResponse.json(
      { error: error?.message || 'Failed to list runs/configs' },
      { status: 500 }
    );
  }
}

