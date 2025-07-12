'use client';

import React, { useEffect, useState } from 'react';

interface SystemStatus {
  vectorCount: number;
  modelVersion: string;
  uptime: string;
}

interface ThreatLog {
  id: string;
  threat: string;
  score: number;
  timestamp: string;
}

export default function DashboardPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [threats, setThreats] = useState<ThreatLog[]>([]);

  useEffect(() => {
    fetch('/api/status')
      .then(res => res.json())
      .then(data => setStatus(data));

    fetch('/api/threats/recent')
      .then(res => res.json())
      .then(data => setThreats(data));
  }, []);

  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Sentenial-X.A.I Dashboard</h1>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">System Status</h2>
        {status ? (
          <div className="bg-gray-100 p-4 rounded">
            <p>🧠 Model: <strong>{status.modelVersion}</strong></p>
            <p>📊 Vector Index Size: <strong>{status.vectorCount}</strong></p>
            <p>⏱️ Uptime: <strong>{status.uptime}</strong></p>
          </div>
        ) : (
          <p>Loading system status...</p>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-2">Recent Threat Events</h2>
        <div className="bg-white shadow rounded p-4">
          {threats.length === 0 ? (
            <p>No recent threats detected.</p>
          ) : (
            <table className="min-w-full text-sm table-auto">
              <thead>
                <tr>
                  <th className="text-left p-2">Time</th>
                  <th className="text-left p-2">Threat</th>
                  <th className="text-left p-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {threats.map(log => (
                  <tr key={log.id} className="border-t">
                    <td className="p-2">{new Date(log.timestamp).toLocaleString()}</td>
                    <td className="p-2">{log.threat}</td>
                    <td className="p-2">{log.score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </main>
  );
}

