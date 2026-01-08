/**
 * API route to upload and import landscape files (.json only)
 */

import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync, readFileSync } from 'fs';
import path from 'path';

import { importNpzData } from '@/lib/import-npz';
import { validateLandscapeData } from '@/lib/validate-landscape';

export const dynamic = 'force-dynamic';

const PROJECT_ROOT = process.env.PROJECT_ROOT || path.join(process.cwd(), '..');
const UPLOAD_DIR = path.join(PROJECT_ROOT, 'uploads');

// Ensure upload directory exists
async function ensureUploadDir() {
  if (!existsSync(UPLOAD_DIR)) {
    await mkdir(UPLOAD_DIR, { recursive: true });
  }
}

export async function POST(request: NextRequest) {
  try {
    await ensureUploadDir();

    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const fileName = file.name.toLowerCase();
    const isJson = fileName.endsWith('.json');

    if (!isJson) {
      return NextResponse.json(
        { error: 'File must be a .json file' },
        { status: 400 }
      );
    }

    // Save uploaded file to temporary location
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const tempFilePath = path.join(UPLOAD_DIR, `${Date.now()}_${file.name}`);
    await writeFile(tempFilePath, buffer);

    try {
      // Read and parse JSON file
      const jsonContent = readFileSync(tempFilePath, 'utf-8');
      // Sanitize NaN/Infinity for JSON parsing
      const sanitizedContent = jsonContent
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');
      
      let data: any;
      try {
        data = JSON.parse(sanitizedContent);
      } catch (parseError: any) {
        return NextResponse.json(
          { error: `Invalid JSON file: ${parseError.message}` },
          { status: 400 }
        );
      }

      // Validate JSON structure
      const validation = validateLandscapeData(data);
      if (!validation.valid) {
        return NextResponse.json(
          {
            error: 'Validation failed',
            validationErrors: validation.errors,
            validationWarnings: validation.warnings,
          },
          { status: 400 }
        );
      }

      // Normalize data format (same as load-from-run)
      data = {
        ...data,
        trajectory_1: data.trajectory_1 || data.trajectory_data?.traj_1,
        trajectory_2: data.trajectory_2 || data.trajectory_data?.traj_2,
        trajectory_3: data.trajectory_3 || data.trajectory_data?.traj_3,
        trajectory_epochs: data.trajectory_epochs || data.trajectory_data?.epochs,
        trajectory_losses: data.trajectory_losses || data.trajectory_data?.losses,
        trajectory_val_losses: data.trajectory_val_losses || data.trajectory_data?.val_losses,
        trajectory_data: data.trajectory_data,
        metadata: data.metadata,
      };

      // Import to database using unified import function
      const sourceLabel = `Uploaded: ${file.name}`;
      const id = await importNpzData(data, sourceLabel, tempFilePath, undefined, undefined, 'upload', file.name);

      return NextResponse.json({
        success: true,
        id,
        data,
        validationWarnings: validateLandscapeData(data).warnings,
      });
    } finally {
      // Clean up temporary file
      try {
        const { unlink } = await import('fs/promises');
        await unlink(tempFilePath);
      } catch (e) {
        console.warn('Failed to delete temp file:', tempFilePath, e);
      }
    }
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

