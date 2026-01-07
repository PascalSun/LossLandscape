/**
 * API route to list all loss landscapes
 */

import { NextRequest, NextResponse } from 'next/server';
import { listLossLandscapes } from '@/lib/db';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const landscapes = await listLossLandscapes();
    return NextResponse.json(landscapes);
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

