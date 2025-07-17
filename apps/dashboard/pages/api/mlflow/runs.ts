// apps/dashboard/pages/api/mlflow/runs.ts

import type { NextApiRequest, NextApiResponse } from "next"
import axios from "axios"

const MLFLOW_URI = process.env.MLFLOW_URI || "http://localhost:5000"

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const experimentId = "0" // or use MLflow's API to get it by name
    const response = await axios.post(`${MLFLOW_URI}/api/2.0/mlflow/runs/search`, {
      experiment_ids: [experimentId],
      max_results: 10,
    });

    const runs = response.data.runs.map((r: any) => {
      const params = Object.fromEntries(r.data.params.map((p: any) => [p.key, p.value]));
      const metrics = Object.fromEntries(r.data.metrics.map((m: any) => [m.key, m.value]));
      return {
        run_id: r.info.run_id,
        accuracy: metrics.cv_accuracy ?? 0,
        c: parseFloat(params.C ?? "1.0"),
        ngram_range: params.ngram_range,
        timestamp: r.info.start_time,
      };
    });

    res.status(200).json({ runs });
  } catch (err) {
    console.error("MLflow API error:", err);
    res.status(500).json({ error: "Failed to fetch MLflow runs" });
  }
}
