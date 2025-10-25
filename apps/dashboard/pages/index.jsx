import { useEffect, useState } from 'react';
import { Container, Typography, Box } from '@mui/material';
import axios from 'axios';
import Header from '../components/Header';

export default function Dashboard() {
  const [threats, setThreats] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/threats', {
      headers: { Authorization: 'Bearer valid-token' }
    }).then(response => {
      setThreats(response.data.threats);
    }).catch(error => {
      console.error('Error fetching threats:', error);
    });
  }, []);

  return (
    <Container>
      <Header />
      <Typography variant="h4" gutterBottom>
        SOC Dashboard
      </Typography>
      <Box>
        {threats.length ? threats.map((threat: any, i: number) => <Typography key={i}>{threat.name}</Typography>) : <Typography>No threats.</Typography>}
      </Box>
    </Container>
  );
}
