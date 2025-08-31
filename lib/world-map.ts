// Real World Map System with Interactive Exploration
export interface WorldTile {
  id: string
  x: number
  y: number
  type: "grass" | "forest" | "mountain" | "water" | "cave" | "village" | "dungeon" | "desert" | "swamp"
  name: string
  description: string
  walkable: boolean
  enemies: WorldEnemy[]
  resources: WorldResource[]
  structures: WorldStructure[]
  discoveryReward?: {
    experience: number
    pi: number
  }
  discovered: boolean
  fogOfWar: boolean
}

export interface WorldEnemy {
  id: string
  name: string
  level: number
  health: number
  maxHealth: number
  attack: number
  defense: number
  experience: number
  loot: any[]
  spawnChance: number
  x: number
  y: number
  isAlive: boolean
  lastSpawnTime: number
}

export interface WorldResource {
  id: string
  name: string
  type: "herb" | "ore" | "crystal" | "treasure"
  quantity: number
  respawnTime: number
  lastHarvested: number
  x: number
  y: number
}

export interface WorldStructure {
  id: string
  name: string
  type: "shop" | "inn" | "temple" | "portal" | "chest" | "npc"
  x: number
  y: number
  interactable: boolean
  discovered: boolean
}

export interface PlayerPosition {
  x: number
  y: number
  facing: "north" | "south" | "east" | "west"
}

export interface MovementResult {
  success: boolean
  newPosition: PlayerPosition
  encounters: EncounterEvent[]
  discoveries: DiscoveryEvent[]
  message?: string
}

export interface EncounterEvent {
  type: "enemy" | "resource" | "structure" | "random"
  data: any
  x: number
  y: number
}

export interface DiscoveryEvent {
  type: "location" | "treasure" | "secret"
  name: string
  reward: {
    experience: number
    pi: number
    items?: any[]
  }
}

class WorldMapEngine {
  private worldSize = 20
  private tiles: Map<string, WorldTile> = new Map()
  private playerPosition: PlayerPosition = { x: 10, y: 10, facing: "north" }
  private visibilityRadius = 2
  private lastMovementTime = 0
  private movementCooldown = 500 // ms between moves

  constructor() {
    this.generateWorld()
    this.spawnInitialEnemies()
  }

  private generateWorld() {
    // Generate a diverse world with different biomes
    for (let x = 0; x < this.worldSize; x++) {
      for (let y = 0; y < this.worldSize; y++) {
        const tileId = `${x}-${y}`
        const tile = this.generateTile(x, y)
        this.tiles.set(tileId, tile)
      }
    }

    // Add special locations
    this.addSpecialLocations()
  }

  private generateTile(x: number, y: number): WorldTile {
    const centerX = this.worldSize / 2
    const centerY = this.worldSize / 2
    const distanceFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)

    let type: WorldTile["type"] = "grass"
    let name = "Plains"
    let description = "Open grasslands with gentle rolling hills"

    // Starting village at center
    if (x === Math.floor(centerX) && y === Math.floor(centerY)) {
      type = "village"
      name = "Mystic Village"
      description = "A peaceful village where your adventure begins"
    }
    // Forest ring around village
    else if (distanceFromCenter > 2 && distanceFromCenter < 6) {
      type = "forest"
      name = "Enchanted Forest"
      description = "Dense woods filled with mysterious creatures"
    }
    // Mountains on edges
    else if (x === 0 || x === this.worldSize - 1 || y === 0 || y === this.worldSize - 1) {
      type = "mountain"
      name = "Mountain Range"
      description = "Towering peaks that touch the clouds"
    }
    // Caves scattered around
    else if (Math.random() < 0.05) {
      type = "cave"
      name = "Crystal Cave"
      description = "A dark cave that glimmers with precious crystals"
    }
    // Water bodies
    else if (Math.random() < 0.08) {
      type = "water"
      name = "Mystic Lake"
      description = "Clear blue waters that reflect the sky"
    }
    // Desert areas
    else if (distanceFromCenter > 8 && Math.random() < 0.3) {
      type = "desert"
      name = "Scorching Sands"
      description = "Hot desert sands stretch endlessly"
    }

    return {
      id: `${x}-${y}`,
      x,
      y,
      type,
      name,
      description,
      walkable: type !== "water" && type !== "mountain",
      enemies: [],
      resources: [],
      structures: [],
      discovered: x === Math.floor(centerX) && y === Math.floor(centerY), // Only starting tile discovered
      fogOfWar: true,
    }
  }

  private addSpecialLocations() {
    // Add dungeons
    const dungeonLocations = [
      { x: 5, y: 5 },
      { x: 15, y: 15 },
      { x: 5, y: 15 },
      { x: 15, y: 5 },
    ]

    dungeonLocations.forEach(({ x, y }) => {
      const tileId = `${x}-${y}`
      const tile = this.tiles.get(tileId)
      if (tile) {
        tile.type = "dungeon"
        tile.name = "Ancient Dungeon"
        tile.description = "A mysterious dungeon filled with treasures and dangers"
        tile.discoveryReward = { experience: 100, pi: 25 }
      }
    })

    // Add shops and inns
    const shopLocations = [
      { x: 8, y: 10, type: "shop" as const },
      { x: 12, y: 10, type: "inn" as const },
      { x: 10, y: 8, type: "temple" as const },
    ]

    shopLocations.forEach(({ x, y, type }) => {
      const tileId = `${x}-${y}`
      const tile = this.tiles.get(tileId)
      if (tile) {
        tile.structures.push({
          id: `${type}-${x}-${y}`,
          name: type === "shop" ? "Merchant's Shop" : type === "inn" ? "Traveler's Inn" : "Sacred Temple",
          type,
          x,
          y,
          interactable: true,
          discovered: false,
        })
      }
    })
  }

  private spawnInitialEnemies() {
    this.tiles.forEach((tile) => {
      if (tile.walkable && tile.type !== "village") {
        this.spawnEnemiesOnTile(tile)
      }
    })
  }

  private spawnEnemiesOnTile(tile: WorldTile) {
    const enemyTypes = this.getEnemyTypesForTile(tile.type)
    const maxEnemies = tile.type === "dungeon" ? 3 : tile.type === "forest" ? 2 : 1

    for (let i = 0; i < maxEnemies; i++) {
      if (Math.random() < 0.3) {
        // 30% chance to spawn enemy
        const enemyType = enemyTypes[Math.floor(Math.random() * enemyTypes.length)]
        const enemy = this.createEnemy(enemyType, tile.x, tile.y)
        tile.enemies.push(enemy)
      }
    }
  }

  private getEnemyTypesForTile(tileType: WorldTile["type"]): string[] {
    switch (tileType) {
      case "forest":
        return ["goblin", "wolf", "spider"]
      case "cave":
        return ["bat", "crystal-golem", "cave-troll"]
      case "desert":
        return ["sand-worm", "desert-bandit", "scorpion"]
      case "dungeon":
        return ["skeleton", "dark-mage", "minotaur"]
      case "swamp":
        return ["swamp-beast", "poison-frog", "will-o-wisp"]
      default:
        return ["slime", "rabbit", "wild-boar"]
    }
  }

  private createEnemy(type: string, x: number, y: number): WorldEnemy {
    const enemyData = {
      goblin: { name: "Forest Goblin", level: 1, health: 30, attack: 8, defense: 2, exp: 15 },
      wolf: { name: "Shadow Wolf", level: 3, health: 50, attack: 12, defense: 4, exp: 25 },
      spider: { name: "Giant Spider", level: 2, health: 35, attack: 10, defense: 3, exp: 20 },
      bat: { name: "Cave Bat", level: 1, health: 20, attack: 6, defense: 1, exp: 10 },
      "crystal-golem": { name: "Crystal Golem", level: 5, health: 80, attack: 15, defense: 8, exp: 50 },
      "cave-troll": { name: "Cave Troll", level: 4, health: 70, attack: 14, defense: 6, exp: 40 },
      "sand-worm": { name: "Sand Worm", level: 3, health: 45, attack: 11, defense: 3, exp: 30 },
      "desert-bandit": { name: "Desert Bandit", level: 4, health: 55, attack: 13, defense: 5, exp: 35 },
      scorpion: { name: "Giant Scorpion", level: 2, health: 40, attack: 9, defense: 4, exp: 22 },
      skeleton: { name: "Ancient Skeleton", level: 6, health: 60, attack: 16, defense: 7, exp: 60 },
      "dark-mage": { name: "Dark Mage", level: 7, health: 50, attack: 20, defense: 5, exp: 80 },
      minotaur: { name: "Minotaur", level: 8, health: 100, attack: 18, defense: 10, exp: 100 },
      slime: { name: "Blue Slime", level: 1, health: 15, attack: 5, defense: 1, exp: 8 },
      rabbit: { name: "Wild Rabbit", level: 1, health: 10, attack: 3, defense: 0, exp: 5 },
      "wild-boar": { name: "Wild Boar", level: 2, health: 25, attack: 7, defense: 2, exp: 12 },
    }

    const data = enemyData[type as keyof typeof enemyData] || enemyData.slime

    return {
      id: `${type}-${x}-${y}-${Date.now()}`,
      name: data.name,
      level: data.level,
      health: data.health,
      maxHealth: data.health,
      attack: data.attack,
      defense: data.defense,
      experience: data.exp,
      loot: [],
      spawnChance: 0.3,
      x,
      y,
      isAlive: true,
      lastSpawnTime: Date.now(),
    }
  }

  getPlayerPosition(): PlayerPosition {
    return { ...this.playerPosition }
  }

  getVisibleTiles(): WorldTile[] {
    const visible: WorldTile[] = []
    const { x: playerX, y: playerY } = this.playerPosition

    for (let x = playerX - this.visibilityRadius; x <= playerX + this.visibilityRadius; x++) {
      for (let y = playerY - this.visibilityRadius; y <= playerY + this.visibilityRadius; y++) {
        const tileId = `${x}-${y}`
        const tile = this.tiles.get(tileId)
        if (tile) {
          // Mark as discovered when player can see it
          if (!tile.discovered) {
            tile.discovered = true
            tile.fogOfWar = false
          }
          visible.push(tile)
        }
      }
    }

    return visible
  }

  getAllDiscoveredTiles(): WorldTile[] {
    return Array.from(this.tiles.values()).filter((tile) => tile.discovered)
  }

  movePlayer(direction: "north" | "south" | "east" | "west"): MovementResult {
    const now = Date.now()
    if (now - this.lastMovementTime < this.movementCooldown) {
      return {
        success: false,
        newPosition: this.playerPosition,
        encounters: [],
        discoveries: [],
        message: "Moving too fast! Wait a moment.",
      }
    }

    const { x, y } = this.playerPosition
    let newX = x
    let newY = y

    switch (direction) {
      case "north":
        newY = Math.max(0, y - 1)
        break
      case "south":
        newY = Math.min(this.worldSize - 1, y + 1)
        break
      case "east":
        newX = Math.min(this.worldSize - 1, x + 1)
        break
      case "west":
        newX = Math.max(0, x - 1)
        break
    }

    const targetTileId = `${newX}-${newY}`
    const targetTile = this.tiles.get(targetTileId)

    if (!targetTile || !targetTile.walkable) {
      return {
        success: false,
        newPosition: this.playerPosition,
        encounters: [],
        discoveries: [],
        message: "Cannot move there!",
      }
    }

    // Update player position
    this.playerPosition = { x: newX, y: newY, facing: direction }
    this.lastMovementTime = now

    // Check for encounters and discoveries
    const encounters: EncounterEvent[] = []
    const discoveries: DiscoveryEvent[] = []

    // Enemy encounters
    const aliveEnemies = targetTile.enemies.filter((enemy) => enemy.isAlive)
    if (aliveEnemies.length > 0 && Math.random() < 0.4) {
      // 40% chance to encounter enemy
      const randomEnemy = aliveEnemies[Math.floor(Math.random() * aliveEnemies.length)]
      encounters.push({
        type: "enemy",
        data: randomEnemy,
        x: newX,
        y: newY,
      })
    }

    // Structure encounters
    targetTile.structures.forEach((structure) => {
      if (structure.interactable) {
        encounters.push({
          type: "structure",
          data: structure,
          x: newX,
          y: newY,
        })
      }
    })

    // Discovery rewards
    if (!targetTile.discovered && targetTile.discoveryReward) {
      discoveries.push({
        type: "location",
        name: targetTile.name,
        reward: targetTile.discoveryReward,
      })
    }

    // Random events
    if (Math.random() < 0.1) {
      // 10% chance for random event
      encounters.push({
        type: "random",
        data: this.generateRandomEvent(targetTile),
        x: newX,
        y: newY,
      })
    }

    return {
      success: true,
      newPosition: this.playerPosition,
      encounters,
      discoveries,
    }
  }

  private generateRandomEvent(tile: WorldTile) {
    const events = [
      {
        type: "treasure",
        message: "You found a hidden treasure chest!",
        reward: { pi: Math.floor(Math.random() * 10) + 5, experience: Math.floor(Math.random() * 20) + 10 },
      },
      {
        type: "trap",
        message: "You stepped on a hidden trap!",
        damage: Math.floor(Math.random() * 15) + 5,
      },
      {
        type: "blessing",
        message: "A mysterious light restores your energy!",
        heal: Math.floor(Math.random() * 30) + 20,
      },
    ]

    return events[Math.floor(Math.random() * events.length)]
  }

  battleEnemy(enemyId: string): { victory: boolean; enemy: WorldEnemy; rewards: any } {
    // Find the enemy across all tiles
    let targetEnemy: WorldEnemy | null = null
    let targetTile: WorldTile | null = null

    for (const tile of this.tiles.values()) {
      const enemy = tile.enemies.find((e) => e.id === enemyId && e.isAlive)
      if (enemy) {
        targetEnemy = enemy
        targetTile = tile
        break
      }
    }

    if (!targetEnemy || !targetTile) {
      return { victory: false, enemy: targetEnemy!, rewards: {} }
    }

    // Mark enemy as dead
    targetEnemy.isAlive = false

    // Simple victory for now - can be expanded with real combat
    const victory = true
    const rewards = {
      experience: targetEnemy.experience,
      pi: Math.floor(targetEnemy.level * 2.5),
      loot: targetEnemy.loot,
    }

    // Respawn enemy after some time
    setTimeout(() => {
      if (targetTile && targetEnemy) {
        const newEnemy = this.createEnemy(
          targetEnemy.name.toLowerCase().replace(/\s+/g, "-"),
          targetTile.x,
          targetTile.y,
        )
        targetTile.enemies.push(newEnemy)
      }
    }, 60000) // Respawn after 1 minute

    return { victory, enemy: targetEnemy, rewards }
  }

  getTileAt(x: number, y: number): WorldTile | null {
    const tileId = `${x}-${y}`
    return this.tiles.get(tileId) || null
  }

  getWorldSize(): number {
    return this.worldSize
  }
}

export const worldMap = new WorldMapEngine()
