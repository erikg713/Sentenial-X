// apps/dashboard/pages/Home.tsx

import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend);

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
  const [retraining, setRetraining] = useState(false);

  const fetchRuns = async () => {
    const res = await fetch("/api/mlflow/runs");
    const data = await res.json();
    setRuns(data.runs);
    setLoading(false);
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const handleRetrain = async () => {
    setRetraining(true);
    const res = await fetch("/api/train", { method: "POST" });
    const result = await res.json();
    alert(result.message || "Retraining completed");
    setRetraining(false);
    fetchRuns(); // Refresh data
  };

  const chartData = {
    labels: runs.map((r) =>
      new Date(r.timestamp).toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
      })
    ),
    datasets: [
      {
        label: "CV Accuracy",
        data: runs.map((r) => r.accuracy * 100),
        borderColor: "#3B82F6",
        backgroundColor: "#93C5FD",
      },
    ],
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-3xl font-bold">📊 Model Training Dashboard</h1>
        <button
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          onClick={handleRetrain}
          disabled={retraining}
        >
          {retraining ? "Retraining..." : "🚀 Retrain Model"}
        </button>
      </div>

      <div className="bg-white shadow p-4 rounded mb-6">
        <h2 className="text-lg font-semibold mb-2">Training Accuracy Over Time</h2>
        <Line data={chartData} />
      </div>

      <div className="overflow-x-auto">
        {loading ? (
          <p>Loading runs…</p>
        ) : (
          <table className="min-w-full table-auto text-left border-collapse border border-gray-300">
            <thead className="bg-gray-100">
              <tr>
                <th className="p-2 border">Run ID</th>
                <th className="p-2 border">Accuracy</th>
                <th className="p-2 border">C</th>
                <th className="p-2 border">N-gram</th>
                <th className="p-2 border">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td className="p-2 border">{run.run_id.slice(0, 8)}…</td>
                  <td className="p-2 border">{(run.accuracy * 100).toFixed(2)}%</td>
                  <td className="p-2 border">{run.c}</td>
                  <td className="p-2 border">{run.ngram_range}</td>
                  <td className="p-2 border">
                    {new Date(run.timestamp).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

