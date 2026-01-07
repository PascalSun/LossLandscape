/**
 * API route to upload and import a .npz file
 */

import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

import { npzToJson } from '@/lib/npz';
import { importNpzData } from '@/lib/import-npz';

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

    if (!file.name.endsWith('.npz')) {
      return NextResponse.json({ error: 'File must be a .npz file' }, { status: 400 });
    }

    // Save uploaded file to temporary location
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const tempFilePath = path.join(UPLOAD_DIR, `${Date.now()}_${file.name}`);
    await writeFile(tempFilePath, buffer);

    try {
      // Convert npz to JSON
      const data = await npzToJson(tempFilePath);

      // Import to database using unified import function
      const sourceLabel = `Uploaded: ${file.name}`;
      const id = await importNpzData(data, sourceLabel);

      return NextResponse.json({ success: true, id, data });
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

