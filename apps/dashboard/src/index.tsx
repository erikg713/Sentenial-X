-<h1>Sentenial-X A.I.</h1>
-<p>The Ultimate Cyber Guardian — Built to Learn, Adapt, and Strike Back</p>
+<header className="hero">
+  <h1>Sentenial-X A.I. 🚀</h1>
+  <h2>The Ultimate Cyber Guardian</h2>
+  <p>Crafted for resilience. Engineered for vengeance.</p>
+  <p>Not just defense — a digital sentinel with the mind of a warrior and the reflexes of a machine.</p>
+</header>

"use client"

import { useState, useEffect } from "react" import { Button } from "@/components/ui/button" import { Card, CardContent, CardTitle } from "@/components/ui/card" import { toast } from "react-hot-toast" import { usePremium } from "@/hooks/usePremium"

export default function Dashboard() { const isPremium = usePremium() const [scanResults, setScanResults] = useState(null)

const handleScan = async () => { toast.loading("Running full security scan...") const res = await fetch("/api/scan", { method: "POST" }) const data = await res.json() toast.dismiss() if (data.issues) { toast.success(🔥 ${data.issues} threats neutralized) setScanResults(data) } else { toast.error("Scan failed or unauthorized.") } }

const handleSubscribe = async () => { const res = await fetch("/api/subscribe", { method: "POST" }) const { payment } = await res.json() if (payment?.url) window.location.href = payment.url else toast.error("Subscription failed.") }

return ( <div className="p-6 space-y-4"> <h1 className="text-3xl font-bold">🛡️ Sentenial X A.I. Dashboard</h1> <div className="grid grid-cols-1 md:grid-cols-2 gap-4"> <Card> <CardContent className="p-4"> <CardTitle>Real-Time Threat Scan</CardTitle> <Button onClick={handleScan} className="mt-2 w-full bg-red-600 text-white"> Run Security Scan </Button> {scanResults && <p className="mt-2">Issues Found: {scanResults.issues}</p>} </CardContent> </Card>

<Card>
      <CardContent className="p-4">
        <CardTitle>🔒 Premium Access</CardTitle>
        {isPremium ? (
          <p className="text-green-600 font-semibold">Active Premium Subscription</p>
        ) : (
          <>
            <p className="mb-2">Upgrade to access advanced modules</p>
            <Button onClick={handleSubscribe} className="bg-yellow-500 hover:bg-yellow-600 w-full">
              Upgrade to Premium 🚀
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  </div>
</div>

) }

