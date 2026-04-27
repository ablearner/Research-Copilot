import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

const MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const segments = (await params).path;
  const filePath = segments.join('/');
  const absPath = path.resolve(process.cwd(), '..', '.data/storage', filePath);

  // Prevent directory traversal
  const storageRoot = path.resolve(process.cwd(), '..', '.data/storage');
  if (!absPath.startsWith(storageRoot)) {
    return new NextResponse('Forbidden', { status: 403 });
  }

  if (!fs.existsSync(absPath)) {
    return new NextResponse('Not found', { status: 404 });
  }

  const ext = path.extname(absPath).toLowerCase();
  const contentType = MIME[ext] || 'application/octet-stream';
  const fileBuffer = fs.readFileSync(absPath);

  return new NextResponse(fileBuffer, {
    headers: {
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=3600',
    },
  });
}
