import Link from "next/link";
import Layout from "../components/Layout";

export default function Home() {
  return (
    <Layout>
      <h1>Sentenial X — Dashboard</h1>
      <p>Welcome — use the login page to get a demo token.</p>
      <p><Link href="/login">Login</Link> | <Link href="/incidents">Incidents</Link></p>
    </Layout>
  );
}