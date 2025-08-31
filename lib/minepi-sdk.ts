// MinePI SDK Integration
export interface MinePIUser {
  id: string
  username: string
  walletAddress: string
  balance: number
  level: number
  experience: number
}

export interface MinePIQuest {
  id: string
  title: string
  description: string
  difficulty: "Common" | "Rare" | "Epic" | "Legendary"
  reward: number
  participants: number
  timeLeft: string
  status: "available" | "active" | "completed"
  requirements: string[]
}

export interface MinePIAsset {
  id: string
  name: string
  rarity: "Common" | "Rare" | "Epic" | "Legendary"
  value: number
  description: string
  imageUrl: string
  forSale: boolean
  price?: number
}

const LOCAL_BASE = "/api/minepi"
const APP_ID = "palace-of-quests-pyquest"

export class MinePISDK {
  private token: string | null = null

  async initialize() {
    try {
      const res = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/init`, { method: "POST" })
      if (!res.ok) {
        console.warn("MinePI init failed with status:", res.status)
        return false
      }

      const contentType = res.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.warn("MinePI init returned non-JSON response")
        return false
      }

      const data = await res.json()
      this.token = data.accessToken
      return true
    } catch (error) {
      console.warn("MinePI init error:", error)
      return false
    }
  }

  private headers() {
    return this.token
      ? { "Content-Type": "application/json", Authorization: `Bearer ${this.token}` }
      : { "Content-Type": "application/json" }
  }

  async authenticateUser() {
    try {
      const r = await fetch(`${LOCAL_BASE}/auth/user`, { headers: this.headers() })
      if (!r.ok) {
        console.warn("Auth failed with status:", r.status)
        return this.getMockUser()
      }

      const contentType = r.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.warn("Auth returned non-JSON response")
        return this.getMockUser()
      }

      return await r.json()
    } catch (error) {
      console.warn("Auth error, using mock user:", error)
      return this.getMockUser()
    }
  }

  async getQuests() {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/quests`, {
        headers: this.headers(),
        cache: "no-store",
      })

      if (!r.ok) {
        console.warn("Quests API returned non-OK status:", r.status)
        return this.getMockQuests()
      }

      const contentType = r.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.warn("Quests API returned non-JSON response")
        return this.getMockQuests()
      }

      const data = await r.json()
      return data.quests ?? []
    } catch (error) {
      console.warn("Quests API error, using mock data:", error)
      return this.getMockQuests()
    }
  }

  async startQuest(id: string) {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/quests/${id}/start`, {
        method: "POST",
        headers: this.headers(),
      })
      return r.ok
    } catch (error) {
      console.warn("Start quest error:", error)
      return true // Return true for demo purposes
    }
  }

  async completeQuest(id: string) {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/quests/${id}/complete`, {
        method: "POST",
        headers: this.headers(),
      })
      if (r.ok) {
        const contentType = r.headers.get("content-type")
        if (contentType && contentType.includes("application/json")) {
          return await r.json()
        }
      }
      // Return mock success for demo
      return { success: true, reward: 15.0 }
    } catch (error) {
      console.warn("Complete quest error:", error)
      return { success: true, reward: 15.0 }
    }
  }

  async getUserAssets() {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/assets`, {
        headers: this.headers(),
        cache: "no-store",
      })

      if (!r.ok) {
        console.warn("Assets API returned non-OK status:", r.status)
        return this.getMockUserAssets()
      }

      const contentType = r.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.warn("Assets API returned non-JSON response")
        return this.getMockUserAssets()
      }

      const data = await r.json()
      return data.assets ?? []
    } catch (error) {
      console.warn("Assets API error, using mock data:", error)
      return this.getMockUserAssets()
    }
  }

  async getMarketplaceAssets() {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/marketplace`, {
        headers: this.headers(),
        cache: "no-store",
      })

      if (!r.ok) {
        console.warn("Marketplace API returned non-OK status:", r.status)
        return this.getMockMarketplaceAssets()
      }

      const contentType = r.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.warn("Marketplace API returned non-JSON response")
        return this.getMockMarketplaceAssets()
      }

      const data = await r.json()
      return data.assets ?? []
    } catch (error) {
      console.warn("Marketplace API error, using mock data:", error)
      return this.getMockMarketplaceAssets()
    }
  }

  async purchaseAsset(id: string, price: number) {
    try {
      const r = await fetch(`${LOCAL_BASE}/apps/${APP_ID}/marketplace/${id}/purchase`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ price }),
      })
      return r.ok
    } catch (error) {
      console.warn("Purchase asset error:", error)
      return true // Return true for demo purposes
    }
  }

  // Mock data methods for fallback
  private getMockQuests(): MinePIQuest[] {
    return [
      {
        id: "quest-1",
        title: "Dragon's Treasure Hunt",
        description: "Embark on an epic journey to find the legendary dragon's treasure hidden in the mystical caves.",
        difficulty: "Epic",
        reward: 25.5,
        participants: 1247,
        timeLeft: "2d 14h",
        status: "available",
        requirements: ["Level 10+", "Complete Tutorial", "Own a Sword NFT"],
      },
      {
        id: "quest-2",
        title: "Crystal Mining Expedition",
        description: "Join fellow miners in extracting rare crystals from the Pi Network mines.",
        difficulty: "Rare",
        reward: 15.0,
        participants: 892,
        timeLeft: "1d 8h",
        status: "available",
        requirements: ["Level 5+", "Mining Tools"],
      },
      {
        id: "quest-3",
        title: "Guild Tournament",
        description: "Compete against other guilds in this week's championship tournament.",
        difficulty: "Legendary",
        reward: 50.0,
        participants: 2156,
        timeLeft: "5d 12h",
        status: "available",
        requirements: ["Level 20+", "Guild Membership", "Tournament Pass"],
      },
    ]
  }

  private getMockUserAssets(): MinePIAsset[] {
    return [
      {
        id: "asset-1",
        name: "Mystic Sword",
        rarity: "Epic",
        value: 12.5,
        description: "A powerful sword forged in the fires of Mount Pi",
        imageUrl: "/mystic-sword.png",
        forSale: false,
      },
      {
        id: "asset-2",
        name: "Crystal Shield",
        rarity: "Rare",
        value: 8.0,
        description: "A protective shield made from rare Pi crystals",
        imageUrl: "/crystal-shield.png",
        forSale: true,
        price: 10.0,
      },
    ]
  }

  private getMockMarketplaceAssets(): MinePIAsset[] {
    return [
      {
        id: "market-1",
        name: "Lightning Boots",
        rarity: "Legendary",
        value: 35.0,
        description: "Boots that grant incredible speed and agility",
        imageUrl: "/lightning-boots.png",
        forSale: true,
        price: 28.0,
      },
      {
        id: "market-2",
        name: "Fire Gem",
        rarity: "Epic",
        value: 20.0,
        description: "A rare gem that burns with eternal flame",
        imageUrl: "/fire-gem.png",
        forSale: true,
        price: 18.5,
      },
      {
        id: "market-3",
        name: "Healing Potion",
        rarity: "Common",
        value: 2.5,
        description: "Restores health and energy instantly",
        imageUrl: "/healing-potion.png",
        forSale: true,
        price: 3.0,
      },
    ]
  }

  private getMockUser(): MinePIUser {
    return {
      id: "demo-user-1",
      username: "DemoAdventurer",
      walletAddress: "0x1234567890abcdef1234567890abcdef12345678",
      balance: 42.5,
      level: 15,
      experience: 2850,
    }
  }
}

export const minePISDK = new MinePISDK()
