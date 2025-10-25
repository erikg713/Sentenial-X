import { useEffect, useState } from 'react';
import { Container, Typography } from '@mui/material';
import axios from 'axios';
import ThreatCard from '../components/ThreatCard';

export default function Threats() {
  const [threats, setThreats] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/threats', {
      headers: { Authorization: 'Bearer valid-token' }
    }).then(res => setThreats(res.data.threats));
  }, []);

  return (
    <Container>
      <Typography variant="h5">Threat List</Typography>
      {threats.map((threat: any, i: number) => <ThreatCard key={i} threat={threat} />)}
    </Container>
  );
}
