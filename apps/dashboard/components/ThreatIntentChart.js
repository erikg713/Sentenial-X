// apps/dashboard/components/ThreatIntentChart.js
import React from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';

const ThreatIntentChart = () => {
  const { data } = useQuery('intents', () => axios.post('/api/cortex/predict', { text: 'Sample log' }));
  return <div>Threat Intent: {data?.intent}</div>;
};
