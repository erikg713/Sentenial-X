import React, { useEffect, useState } from "react";
import axios from "axios";

interface Threat {
  id: string;
  title: string;
  description: string;
  timestamp: string; // ISO string
  severity: "Low" | "Medium" | "High" | "Critical";
}

const severityColorMap: Record<Threat["severity"], string> = {
  Low: "text-green-500",
  Medium: "text-yellow-400",
  High: "text-red-500",
  Critical: "text-red-700 font-bold",
};

const ThreatFeed: React.FC = () => {
  const [threats, setThreats] = useState<Threat[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchThreats = async () => {
      try {
        setLoading(true);
        const response = await axios.get<Threat[]>("/api/threats");
        setThreats(response.data);
      } catch (err) {
        setError("Failed to fetch threat data.");
      } finally {
        setLoading(false);
      }
    };

    fetchThreats();

    // Optional: Poll every 60 seconds
    const interval = setInterval(fetchThreats, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div className="p-4">Loading threats...</div>;
  if (error) return <div className="p-4 text-red-600">{error}</div>;
  if (threats.length === 0) return <div className="p-4">No threat data available.</div>;

  return (
    <div className="max-h-96 overflow-y-auto bg-gray-800 text-white rounded shadow-md p-4">
      <h2 className="text-xl font-semibold mb-4">Threat Feed</h2>
      <ul>
        {threats.map(({ id, title, description, timestamp, severity }) => (
          <li
            key={id}
            className="border-b border-gray-700 py-2 last:border-b-0"
            title={description}
          >
            <div className="flex justify-between items-center">
              <span className={`text-sm font-semibold ${severityColorMap[severity]}`}>
                [{severity}]
              </span>
              <time
                className="text-xs text-gray-400"
                dateTime={timestamp}
                title={new Date(timestamp).toLocaleString()}
              >
                {new Date(timestamp).toLocaleTimeString()}
              </time>
            </div>
            <p className="text-md mt-1">{title}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ThreatFeed;
