// apps/dashboard/pages/Home.tsx

import { useEffect, useState } from "react";

type RunMetric = {
  run_id: string;
  accuracy: number;
  c: number;
  ngram_range: string;
  timestamp: string;
};

export default function Home() {
  const [runs, setRuns] = useState<RunMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/mlflow/runs") // <- this should be your proxy endpoint
      .then((res) => res.json())
      .then((data) => {
        setRuns(data.runs);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-4">📊 Model Training Dashboard</h1>

      {loading ? (
        <p>Loading runs...</p>
      ) : (
        <table className="table-auto w-full text-left border">
          <thead>
            <tr className="bg-gray-100">
              <th className="p-2">Run ID</th>
              <th className="p-2">Accuracy</th>
              <th className="p-2">C</th>
              <th className="p-2">N-gram</th>
              <th className="p-2">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td className="p-2">{run.run_id.slice(0, 8)}…</td>
                <td className="p-2">{(run.accuracy * 100).toFixed(2)}%</td>
                <td className="p-2">{run.c}</td>
                <td className="p-2">{run.ngram_range}</td>
                <td className="p-2">{new Date(run.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

