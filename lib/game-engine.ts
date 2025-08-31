// Real Game Engine - No Demo Data
export interface GameUser {
  id: string
  username: string
  walletAddress: string
  balance: number
  level: number
  experience: number
  health: number
  maxHealth: number
  energy: number
  maxEnergy: number
  stats: {
    strength: number
    defense: number
    agility: number
    intelligence: number
  }
  inventory: GameItem[]
  activeQuests: string[]
  completedQuests: string[]
  location: string
}

export interface GameQuest {
  id: string
  title: string
  description: string
  difficulty: "Common" | "Rare" | "Epic" | "Legendary"
  reward: {
    pi: number
    experience: number
    items?: GameItem[]
  }
  requirements: {
    level?: number
    items?: string[]
    completedQuests?: string[]
  }
  objectives: QuestObjective[]
  timeLimit?: number
  status: "available" | "active" | "completed" | "failed"
  progress: number
}

export interface QuestObjective {
  id: string
  description: string
  type: "kill" | "collect" | "explore" | "craft" | "talk"
  target: string
  current: number
  required: number
  completed: boolean
}

export interface GameItem {
  id: string
  name: string
  type: "weapon" | "armor" | "consumable" | "material" | "treasure"
  rarity: "Common" | "Rare" | "Epic" | "Legendary"
  value: number
  description: string
  stats?: {
    attack?: number
    defense?: number
    health?: number
    energy?: number
  }
  quantity: number
  equipped?: boolean
}

export interface Enemy {
  id: string
  name: string
  level: number
  health: number
  maxHealth: number
  attack: number
  defense: number
  experience: number
  loot: GameItem[]
}

export interface GameLocation {
  id: string
  name: string
  description: string
  enemies: Enemy[]
  resources: GameItem[]
  quests: string[]
  unlockRequirements?: {
    level?: number
    completedQuests?: string[]
  }
}

class GameEngine {
  private user: GameUser | null = null
  private locations: Map<string, GameLocation> = new Map()
  private quests: Map<string, GameQuest> = new Map()
  private enemies: Map<string, Enemy> = new Map()
  private items: Map<string, GameItem> = new Map()

  constructor() {
    this.initializeGameData()
  }

  private initializeGameData() {
    // Initialize game locations
    this.locations.set("village", {
      id: "village",
      name: "Mystic Village",
      description: "A peaceful village where your adventure begins",
      enemies: [],
      resources: [],
      quests: ["tutorial", "first-hunt"],
    })

    this.locations.set("forest", {
      id: "forest",
      name: "Enchanted Forest",
      description: "A mysterious forest filled with magical creatures",
      enemies: ["goblin", "wolf"],
      resources: [],
      quests: ["forest-exploration"],
      unlockRequirements: { level: 2 },
    })

    this.locations.set("cave", {
      id: "cave",
      name: "Crystal Caves",
      description: "Deep caves containing precious crystals and dangerous monsters",
      enemies: ["bat", "spider", "crystal-golem"],
      resources: [],
      quests: ["crystal-mining"],
      unlockRequirements: { level: 5 },
    })

    // Initialize enemies
    this.enemies.set("goblin", {
      id: "goblin",
      name: "Forest Goblin",
      level: 1,
      health: 30,
      maxHealth: 30,
      attack: 8,
      defense: 2,
      experience: 15,
      loot: [
        {
          id: "goblin-tooth",
          name: "Goblin Tooth",
          type: "material",
          rarity: "Common",
          value: 2,
          description: "A sharp tooth from a forest goblin",
          quantity: 1,
        },
      ],
    })

    this.enemies.set("wolf", {
      id: "wolf",
      name: "Shadow Wolf",
      level: 3,
      health: 50,
      maxHealth: 50,
      attack: 12,
      defense: 4,
      experience: 25,
      loot: [
        {
          id: "wolf-pelt",
          name: "Wolf Pelt",
          type: "material",
          rarity: "Rare",
          value: 8,
          description: "Soft fur from a shadow wolf",
          quantity: 1,
        },
      ],
    })

    // Initialize items
    this.items.set("health-potion", {
      id: "health-potion",
      name: "Health Potion",
      type: "consumable",
      rarity: "Common",
      value: 5,
      description: "Restores 50 health points",
      stats: { health: 50 },
      quantity: 1,
    })

    this.items.set("iron-sword", {
      id: "iron-sword",
      name: "Iron Sword",
      type: "weapon",
      rarity: "Common",
      value: 25,
      description: "A sturdy iron sword",
      stats: { attack: 15 },
      quantity: 1,
    })

    // Initialize quests
    this.quests.set("tutorial", {
      id: "tutorial",
      title: "Welcome to Palace of Quests",
      description: "Learn the basics of adventuring in this mystical world",
      difficulty: "Common",
      reward: { pi: 5, experience: 50 },
      requirements: {},
      objectives: [
        {
          id: "explore-village",
          description: "Explore the village",
          type: "explore",
          target: "village",
          current: 0,
          required: 1,
          completed: false,
        },
      ],
      status: "available",
      progress: 0,
    })

    this.quests.set("first-hunt", {
      id: "first-hunt",
      title: "First Hunt",
      description: "Defeat your first enemy to prove your worth as an adventurer",
      difficulty: "Common",
      reward: { pi: 10, experience: 100, items: [this.items.get("health-potion")!] },
      requirements: { level: 1 },
      objectives: [
        {
          id: "kill-goblin",
          description: "Defeat a Forest Goblin",
          type: "kill",
          target: "goblin",
          current: 0,
          required: 1,
          completed: false,
        },
      ],
      status: "available",
      progress: 0,
    })
  }

  createUser(username: string): GameUser {
    this.user = {
      id: `user_${Date.now()}`,
      username,
      walletAddress: `0x${Math.random().toString(16).substr(2, 40)}`,
      balance: 0,
      level: 1,
      experience: 0,
      health: 100,
      maxHealth: 100,
      energy: 50,
      maxEnergy: 50,
      stats: {
        strength: 10,
        defense: 8,
        agility: 6,
        intelligence: 5,
      },
      inventory: [
        { ...this.items.get("health-potion")!, quantity: 3 },
        { ...this.items.get("iron-sword")!, equipped: true },
      ],
      activeQuests: [],
      completedQuests: [],
      location: "village",
    }
    return this.user
  }

  getUser(): GameUser | null {
    return this.user
  }

  getAvailableQuests(): GameQuest[] {
    if (!this.user) return []

    const currentLocation = this.locations.get(this.user.location)
    if (!currentLocation) return []

    return currentLocation.quests
      .map((questId) => this.quests.get(questId))
      .filter((quest): quest is GameQuest => {
        if (!quest) return false
        if (quest.status !== "available") return false
        if (this.user!.completedQuests.includes(quest.id)) return false
        if (this.user!.activeQuests.includes(quest.id)) return false

        // Check requirements
        if (quest.requirements.level && this.user!.level < quest.requirements.level) return false
        if (quest.requirements.completedQuests) {
          for (const reqQuest of quest.requirements.completedQuests) {
            if (!this.user!.completedQuests.includes(reqQuest)) return false
          }
        }

        return true
      })
  }

  getActiveQuests(): GameQuest[] {
    if (!this.user) return []

    return this.user.activeQuests
      .map((questId) => this.quests.get(questId))
      .filter((quest): quest is GameQuest => quest !== undefined)
  }

  startQuest(questId: string): boolean {
    if (!this.user) return false

    const quest = this.quests.get(questId)
    if (!quest || quest.status !== "available") return false

    quest.status = "active"
    this.user.activeQuests.push(questId)
    return true
  }

  updateQuestProgress(questId: string, objectiveId: string, amount = 1): void {
    if (!this.user) return

    const quest = this.quests.get(questId)
    if (!quest || quest.status !== "active") return

    const objective = quest.objectives.find((obj) => obj.id === objectiveId)
    if (!objective || objective.completed) return

    objective.current = Math.min(objective.current + amount, objective.required)
    objective.completed = objective.current >= objective.required

    // Update quest progress
    const completedObjectives = quest.objectives.filter((obj) => obj.completed).length
    quest.progress = (completedObjectives / quest.objectives.length) * 100

    // Check if quest is complete
    if (quest.progress >= 100) {
      this.completeQuest(questId)
    }
  }

  completeQuest(questId: string): boolean {
    if (!this.user) return false

    const quest = this.quests.get(questId)
    if (!quest || quest.status !== "active") return false

    quest.status = "completed"
    this.user.activeQuests = this.user.activeQuests.filter((id) => id !== questId)
    this.user.completedQuests.push(questId)

    // Give rewards
    this.user.balance += quest.reward.pi
    this.addExperience(quest.reward.experience)

    if (quest.reward.items) {
      for (const item of quest.reward.items) {
        this.addItemToInventory(item)
      }
    }

    return true
  }

  addExperience(amount: number): void {
    if (!this.user) return

    this.user.experience += amount

    // Check for level up
    const requiredExp = this.user.level * 100
    if (this.user.experience >= requiredExp) {
      this.levelUp()
    }
  }

  private levelUp(): void {
    if (!this.user) return

    this.user.level++
    this.user.experience = 0
    this.user.maxHealth += 20
    this.user.maxEnergy += 10
    this.user.health = this.user.maxHealth
    this.user.energy = this.user.maxEnergy

    // Increase stats
    this.user.stats.strength += 2
    this.user.stats.defense += 2
    this.user.stats.agility += 1
    this.user.stats.intelligence += 1
  }

  addItemToInventory(item: GameItem): void {
    if (!this.user) return

    const existingItem = this.user.inventory.find((invItem) => invItem.id === item.id)
    if (existingItem) {
      existingItem.quantity += item.quantity
    } else {
      this.user.inventory.push({ ...item })
    }
  }

  battle(enemyId: string): { victory: boolean; rewards: GameItem[]; experience: number } {
    if (!this.user) return { victory: false, rewards: [], experience: 0 }

    const enemyTemplate = this.enemies.get(enemyId)
    if (!enemyTemplate) return { victory: false, rewards: [], experience: 0 }

    const enemy = { ...enemyTemplate }

    // Simple battle calculation
    const playerAttack = this.user.stats.strength + (this.getEquippedWeapon()?.stats?.attack || 0)
    const playerDefense = this.user.stats.defense + (this.getEquippedArmor()?.stats?.defense || 0)

    const playerDamage = Math.max(1, playerAttack - enemy.defense)
    const enemyDamage = Math.max(1, enemy.attack - playerDefense)

    // Battle simulation
    while (this.user.health > 0 && enemy.health > 0) {
      enemy.health -= playerDamage
      if (enemy.health <= 0) break

      this.user.health -= enemyDamage
    }

    const victory = enemy.health <= 0

    if (victory) {
      this.addExperience(enemy.experience)

      // Update quest progress for kill objectives
      for (const questId of this.user.activeQuests) {
        const quest = this.quests.get(questId)
        if (quest) {
          const killObjective = quest.objectives.find((obj) => obj.type === "kill" && obj.target === enemyId)
          if (killObjective) {
            this.updateQuestProgress(questId, killObjective.id)
          }
        }
      }

      return { victory: true, rewards: enemy.loot, experience: enemy.experience }
    } else {
      // Player died, respawn with reduced health
      this.user.health = Math.floor(this.user.maxHealth * 0.5)
      return { victory: false, rewards: [], experience: 0 }
    }
  }

  private getEquippedWeapon(): GameItem | undefined {
    return this.user?.inventory.find((item) => item.type === "weapon" && item.equipped)
  }

  private getEquippedArmor(): GameItem | undefined {
    return this.user?.inventory.find((item) => item.type === "armor" && item.equipped)
  }

  useItem(itemId: string): boolean {
    if (!this.user) return false

    const item = this.user.inventory.find((invItem) => invItem.id === itemId)
    if (!item || item.quantity <= 0) return false

    if (item.type === "consumable") {
      if (item.stats?.health) {
        this.user.health = Math.min(this.user.maxHealth, this.user.health + item.stats.health)
      }
      if (item.stats?.energy) {
        this.user.energy = Math.min(this.user.maxEnergy, this.user.energy + item.stats.energy)
      }

      item.quantity--
      if (item.quantity <= 0) {
        this.user.inventory = this.user.inventory.filter((invItem) => invItem.id !== itemId)
      }

      return true
    }

    return false
  }

  getLocations(): GameLocation[] {
    if (!this.user) return []

    return Array.from(this.locations.values()).filter((location) => {
      if (!location.unlockRequirements) return true

      if (location.unlockRequirements.level && this.user!.level < location.unlockRequirements.level) {
        return false
      }

      if (location.unlockRequirements.completedQuests) {
        for (const reqQuest of location.unlockRequirements.completedQuests) {
          if (!this.user!.completedQuests.includes(reqQuest)) return false
        }
      }

      return true
    })
  }

  changeLocation(locationId: string): boolean {
    if (!this.user) return false

    const location = this.locations.get(locationId)
    if (!location) return false

    // Check unlock requirements
    if (location.unlockRequirements) {
      if (location.unlockRequirements.level && this.user.level < location.unlockRequirements.level) {
        return false
      }

      if (location.unlockRequirements.completedQuests) {
        for (const reqQuest of location.unlockRequirements.completedQuests) {
          if (!this.user.completedQuests.includes(reqQuest)) return false
        }
      }
    }

    this.user.location = locationId

    // Update quest progress for explore objectives
    for (const questId of this.user.activeQuests) {
      const quest = this.quests.get(questId)
      if (quest) {
        const exploreObjective = quest.objectives.find((obj) => obj.type === "explore" && obj.target === locationId)
        if (exploreObjective) {
          this.updateQuestProgress(questId, exploreObjective.id)
        }
      }
    }

    return true
  }

  getCurrentLocation(): GameLocation | null {
    if (!this.user) return null
    return this.locations.get(this.user.location) || null
  }

  getEnemiesInCurrentLocation(): Enemy[] {
    const location = this.getCurrentLocation()
    if (!location) return []

    return location.enemies
      .map((enemyId) => this.enemies.get(enemyId))
      .filter((enemy): enemy is Enemy => enemy !== undefined)
  }

  restoreHealth(): void {
    if (!this.user) return
    this.user.health = this.user.maxHealth
    this.user.energy = this.user.maxEnergy
  }
}

export const gameEngine = new GameEngine()
