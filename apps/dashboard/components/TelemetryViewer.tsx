import React, { useEffect, useState, useMemo } from "react";
import axios from "axios";

interface TelemetryEntry {
  id: string;
  timestamp: string; // ISO date string
  metric: string;
  value: number | string;
  source: string;
}

const TelemetryViewer: React.FC = () => {
  const [data, setData] = useState<TelemetryEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        setLoading(true);
        const res = await axios.get<TelemetryEntry[]>("/api/telemetry");
        setData(res.data);
        setError(null);
      } catch (err) {
        setError("Failed to load telemetry data.");
      } finally {
        setLoading(false);
      }
    };

    fetchTelemetry();

    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchTelemetry, 30000);
    return () => clearInterval(interval);
  }, []);

  const filteredData = useMemo(() => {
    if (!searchTerm) return data;
    return data.filter(
      (entry) =>
        entry.metric.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.source.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [data, searchTerm]);

  if (loading) return <div className="p-4">Loading telemetry...</div>;
  if (error) return <div className="p-4 text-red-600">{error}</div>;
  if (filteredData.length === 0)
    return <div className="p-4">No telemetry data found.</div>;

  return (
    <div className="p-4 bg-gray-900 rounded shadow-md text-white max-w-full overflow-x-auto">
      <h2 className="text-xl font-semibold mb-4">Telemetry Viewer</h2>

      <input
        type="text"
        placeholder="Search metric or source..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="mb-4 px-3 py-2 rounded text-black w-full max-w-sm"
      />

      <table className="min-w-full table-auto border-collapse border border-gray-700">
        <thead>
          <tr className="bg-gray-800">
            <th className="border border-gray-600 px-4 py-2 text-left">Timestamp</th>
            <th className="border border-gray-600 px-4 py-2 text-left">Metric</th>
            <th className="border border-gray-600 px-4 py-2 text-left">Value</th>
            <th className="border border-gray-600 px-4 py-2 text-left">Source</th>
          </tr>
        </thead>
        <tbody>
          {filteredData.map(({ id, timestamp, metric, value, source }) => (
            <tr
              key={id}
              className="hover:bg-gray-700 cursor-default"
              title={`${metric} from ${source} at ${new Date(timestamp).toLocaleString()}`}
            >
              <td className="border border-gray-600 px-4 py-2">
                {new Date(timestamp).toLocaleString()}
              </td>
              <td className="border border-gray-600 px-4 py-2">{metric}</td>
              <td className="border border-gray-600 px-4 py-2">{value}</td>
              <td className="border border-gray-600 px-4 py-2">{source}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TelemetryViewer;


