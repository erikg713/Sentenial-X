import { useEffect, useState } from "react"

export function usePremium() {
  const [isPremium, setIsPremium] = useState(false)

  useEffect(() => {
    fetch("/api/premium-status")
      .then((res) => res.json())
      .then((data) => setIsPremium(data.status === "active"))
  }, [])

  return isPremium
}
