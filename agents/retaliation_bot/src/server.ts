import express from 'express';
import cors from 'cors';
import { runReconAnalysis } from './recon';

const app = express();
app.use(cors());
app.use(express.json());

// Endpoint: /api/recon?location=London
app.get('/api/recon', async (req, res) => {
  const { location } = req.query as { location?: string };

  if (!location) {
    return res.status(400).json({ error: 'Missing location parameter' });
  }

  try {
    // runReconAnalysis writes to console; adapt it to return JSON
    const result = await runReconAnalysis(location);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`API listening on http://localhost:${PORT}`);
});
