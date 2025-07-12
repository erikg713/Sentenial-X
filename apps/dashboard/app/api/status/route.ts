import { NextResponse } from 'next/server';

export async function GET() {
  // Simulated system metrics
  const data = {
    modelVersion: '1.1.0-finetuned',
    vectorCount: 1284,
    uptime: '3 days, 12 hours',
  };

  return NextResponse.json(data);
}
