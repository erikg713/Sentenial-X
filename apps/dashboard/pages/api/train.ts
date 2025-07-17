// apps/dashboard/pages/api/train.ts

import type { NextApiRequest, NextApiResponse } from "next";
import { exec } from "child_process";
import path from "path";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const feedbackPath = path.resolve(process.cwd(), "data", "feedback.json");

  // Modify this to match your actual CLI entrypoint if needed
  const command = `python3 -m ml_pipeline.train ${feedbackPath}`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error("Retraining error:", stderr);
      return res.status(500).json({ error: "Training failed" });
    }

    console.log("Retraining output:", stdout);
    return res.status(200).json({ message: "Retraining completed" });
  });
}
