import { NextResponse } from 'next/server';

export async function GET() {
  // Simulated recent threat log
  const data = [
    {
      id: 'threat-001',
      threat: 'sql injection',
      score: 0.92,
      timestamp: new Date().toISOString(),
    },
    {
      id: 'threat-002',
      threat: 'prompt injection',
      score: 0.89,
      timestamp: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
    },
    {
      id: 'threat-003',
      threat: 'command injection',
      score: 0.86,
      timestamp: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
    },
  ];

  return NextResponse.json(data);
}
