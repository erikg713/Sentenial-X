import { Card, CardContent, Typography } from '@mui/material';

interface ThreatProps {
  threat: { name: string; score: number };
}

export default function ThreatCard({ threat }: ThreatProps) {
  return (
    <Card>
      <CardContent>
        <Typography>{threat.name}</Typography>
        <Typography>Score: {threat.score}</Typography>
      </CardContent>
    </Card>
  );
}
