// Web3 Metaverse Character System with Deep RPG Mechanics
export interface CharacterClass {
  id: string
  name: string
  description: string
  startingStats: CharacterStats
  abilities: ClassAbility[]
  equipment: string[]
  lore: string
}

export interface CharacterStats {
  strength: number
  dexterity: number
  intelligence: number
  wisdom: number
  constitution: number
  charisma: number
  // Web3 specific stats
  miningPower: number
  tradingSkill: number
  socialInfluence: number
}

export interface ClassAbility {
  id: string
  name: string
  description: string
  cooldown: number
  manaCost: number
  effect: AbilityEffect
}

export interface AbilityEffect {
  type: "damage" | "heal" | "buff" | "debuff" | "utility" | "social"
  value: number
  duration?: number
  target: "self" | "enemy" | "ally" | "area"
}

export interface CharacterBackground {
  id: string
  name: string
  description: string
  bonuses: Partial<CharacterStats>
  startingItems: string[]
  relationships: NPCRelationship[]
  lore: string
}

export interface NPCRelationship {
  npcId: string
  relationship: "friend" | "neutral" | "enemy" | "romantic" | "rival" | "mentor"
  level: number // -100 to 100
}

export interface DialogueOption {
  id: string
  text: string
  requirements?: {
    stats?: Partial<CharacterStats>
    items?: string[]
    relationships?: NPCRelationship[]
    class?: string
    background?: string
  }
  consequences: DialogueConsequence[]
}

export interface DialogueConsequence {
  type: "relationship" | "item" | "stat" | "quest" | "reputation" | "pi"
  target: string
  value: number
  message: string
}

export interface MetaverseReputation {
  guild: string
  level: number
  title: string
  benefits: string[]
}

export interface Web3Asset {
  id: string
  name: string
  type: "nft" | "token" | "land" | "building" | "vehicle"
  rarity: "Common" | "Rare" | "Epic" | "Legendary" | "Mythic"
  attributes: Record<string, any>
  owner: string
  tradeable: boolean
  stakeable: boolean
  utility: string[]
  metaverseValue: number
}

const CHARACTER_CLASSES: CharacterClass[] = [
  {
    id: "paladin",
    name: "Pi Paladin",
    description: "A noble warrior who protects the Pi Network and upholds justice in the metaverse",
    startingStats: {
      strength: 16,
      dexterity: 12,
      intelligence: 13,
      wisdom: 15,
      constitution: 14,
      charisma: 14,
      miningPower: 8,
      tradingSkill: 6,
      socialInfluence: 12,
    },
    abilities: [
      {
        id: "divine-shield",
        name: "Divine Shield",
        description: "Protects allies with the power of Pi",
        cooldown: 30,
        manaCost: 20,
        effect: { type: "buff", value: 50, duration: 60, target: "ally" },
      },
      {
        id: "righteous-strike",
        name: "Righteous Strike",
        description: "A powerful attack that deals extra damage to corrupt entities",
        cooldown: 15,
        manaCost: 15,
        effect: { type: "damage", value: 120, target: "enemy" },
      },
    ],
    equipment: ["holy-sword", "blessed-armor", "pi-amulet"],
    lore: "Paladins are the guardians of the Pi Network, sworn to protect honest miners and traders from malicious actors.",
  },
  {
    id: "crypto-mage",
    name: "Crypto Mage",
    description: "A master of blockchain magic who manipulates smart contracts and digital energy",
    startingStats: {
      strength: 8,
      dexterity: 12,
      intelligence: 18,
      wisdom: 16,
      constitution: 10,
      charisma: 12,
      miningPower: 15,
      tradingSkill: 14,
      socialInfluence: 8,
    },
    abilities: [
      {
        id: "smart-contract",
        name: "Smart Contract",
        description: "Creates magical contracts that bind enemies",
        cooldown: 25,
        manaCost: 30,
        effect: { type: "debuff", value: -30, duration: 45, target: "enemy" },
      },
      {
        id: "pi-lightning",
        name: "Pi Lightning",
        description: "Channels the power of the blockchain into devastating lightning",
        cooldown: 20,
        manaCost: 25,
        effect: { type: "damage", value: 150, target: "area" },
      },
    ],
    equipment: ["staff-of-satoshi", "robes-of-encryption", "mining-crystal"],
    lore: "Crypto Mages study the ancient arts of blockchain manipulation, wielding the fundamental forces of decentralization.",
  },
  {
    id: "nft-hunter",
    name: "NFT Hunter",
    description: "A skilled tracker and collector who specializes in finding rare digital assets",
    startingStats: {
      strength: 14,
      dexterity: 18,
      intelligence: 14,
      wisdom: 13,
      constitution: 12,
      charisma: 11,
      miningPower: 6,
      tradingSkill: 18,
      socialInfluence: 10,
    },
    abilities: [
      {
        id: "asset-scan",
        name: "Asset Scan",
        description: "Reveals hidden NFTs and their true value",
        cooldown: 10,
        manaCost: 10,
        effect: { type: "utility", value: 1, target: "area" },
      },
      {
        id: "precision-strike",
        name: "Precision Strike",
        description: "A perfectly aimed attack that never misses",
        cooldown: 12,
        manaCost: 12,
        effect: { type: "damage", value: 100, target: "enemy" },
      },
    ],
    equipment: ["hunters-bow", "tracking-boots", "collectors-bag"],
    lore: "NFT Hunters roam the metaverse seeking rare and valuable digital assets, masters of appraisal and acquisition.",
  },
  {
    id: "defi-merchant",
    name: "DeFi Merchant",
    description: "A charismatic trader who builds wealth through decentralized finance",
    startingStats: {
      strength: 10,
      dexterity: 14,
      intelligence: 15,
      wisdom: 12,
      constitution: 11,
      charisma: 18,
      miningPower: 5,
      tradingSkill: 20,
      socialInfluence: 16,
    },
    abilities: [
      {
        id: "market-manipulation",
        name: "Market Insight",
        description: "Predicts market movements and adjusts prices",
        cooldown: 60,
        manaCost: 20,
        effect: { type: "utility", value: 200, target: "self" },
      },
      {
        id: "charm-customer",
        name: "Charm Customer",
        description: "Convinces others to make favorable trades",
        cooldown: 30,
        manaCost: 15,
        effect: { type: "social", value: 50, target: "ally" },
      },
    ],
    equipment: ["golden-ledger", "merchants-robes", "persuasion-ring"],
    lore: "DeFi Merchants understand the flow of digital wealth, building empires through smart trading and social connections.",
  },
]

const CHARACTER_BACKGROUNDS: CharacterBackground[] = [
  {
    id: "early-adopter",
    name: "Early Adopter",
    description: "You were among the first to join the Pi Network, giving you deep knowledge and connections",
    bonuses: { miningPower: 5, socialInfluence: 3, wisdom: 2 },
    startingItems: ["vintage-pi-coin", "founders-badge", "network-map"],
    relationships: [
      { npcId: "pi-founder", relationship: "friend", level: 50 },
      { npcId: "veteran-miner", relationship: "mentor", level: 30 },
    ],
    lore: "Early Adopters remember the beginning of Pi Network and are respected for their foresight and dedication.",
  },
  {
    id: "digital-nomad",
    name: "Digital Nomad",
    description: "You've traveled the digital realms extensively, gaining diverse experiences",
    bonuses: { dexterity: 3, intelligence: 2, tradingSkill: 4 },
    startingItems: ["travel-pack", "universal-translator", "portal-key"],
    relationships: [
      { npcId: "portal-keeper", relationship: "friend", level: 40 },
      { npcId: "realm-guide", relationship: "neutral", level: 20 },
    ],
    lore: "Digital Nomads have seen many virtual worlds and bring unique perspectives to the Pi metaverse.",
  },
  {
    id: "guild-exile",
    name: "Guild Exile",
    description: "Cast out from a powerful guild, you now forge your own path with determination",
    bonuses: { strength: 3, constitution: 3, charisma: -2 },
    startingItems: ["broken-guild-seal", "exile-cloak", "vendetta-blade"],
    relationships: [
      { npcId: "guild-master", relationship: "enemy", level: -60 },
      { npcId: "fellow-exile", relationship: "friend", level: 70 },
    ],
    lore: "Guild Exiles carry the weight of betrayal but have learned to rely on their own strength and cunning.",
  },
  {
    id: "ai-whisperer",
    name: "AI Whisperer",
    description: "You have a unique ability to communicate with AI entities in the metaverse",
    bonuses: { intelligence: 4, wisdom: 3, socialInfluence: 2 },
    startingItems: ["ai-interface", "neural-crown", "data-crystal"],
    relationships: [
      { npcId: "ai-oracle", relationship: "friend", level: 80 },
      { npcId: "tech-priest", relationship: "mentor", level: 45 },
    ],
    lore: "AI Whisperers bridge the gap between human consciousness and artificial intelligence.",
  },
]

export class CharacterCreationSystem {
  private selectedClass: CharacterClass | null = null
  private selectedBackground: CharacterBackground | null = null
  private customStats: Partial<CharacterStats> = {}
  private appearance: CharacterAppearance = {
    skinTone: "medium",
    hairColor: "brown",
    eyeColor: "brown",
    height: "average",
    build: "athletic",
    markings: [],
    clothing: "default",
  }

  getAvailableClasses(): CharacterClass[] {
    return CHARACTER_CLASSES
  }

  getAvailableBackgrounds(): CharacterBackground[] {
    return CHARACTER_BACKGROUNDS
  }

  selectClass(classId: string): boolean {
    const characterClass = CHARACTER_CLASSES.find((c) => c.id === classId)
    if (characterClass) {
      this.selectedClass = characterClass
      return true
    }
    return false
  }

  selectBackground(backgroundId: string): boolean {
    const background = CHARACTER_BACKGROUNDS.find((b) => b.id === backgroundId)
    if (background) {
      this.selectedBackground = background
      return true
    }
    return false
  }

  setAppearance(appearance: Partial<CharacterAppearance>): void {
    this.appearance = { ...this.appearance, ...appearance }
  }

  calculateFinalStats(): CharacterStats {
    if (!this.selectedClass) {
      throw new Error("No class selected")
    }

    const baseStats = { ...this.selectedClass.startingStats }
    const backgroundBonuses = this.selectedBackground?.bonuses || {}
    const customBonuses = this.customStats

    // Combine all stat sources
    const finalStats: CharacterStats = {
      strength: (baseStats.strength || 0) + (backgroundBonuses.strength || 0) + (customBonuses.strength || 0),
      dexterity: (baseStats.dexterity || 0) + (backgroundBonuses.dexterity || 0) + (customBonuses.dexterity || 0),
      intelligence:
        (baseStats.intelligence || 0) + (backgroundBonuses.intelligence || 0) + (customBonuses.intelligence || 0),
      wisdom: (baseStats.wisdom || 0) + (backgroundBonuses.wisdom || 0) + (customBonuses.wisdom || 0),
      constitution:
        (baseStats.constitution || 0) + (backgroundBonuses.constitution || 0) + (customBonuses.constitution || 0),
      charisma: (baseStats.charisma || 0) + (backgroundBonuses.charisma || 0) + (customBonuses.charisma || 0),
      miningPower:
        (baseStats.miningPower || 0) + (backgroundBonuses.miningPower || 0) + (customBonuses.miningPower || 0),
      tradingSkill:
        (baseStats.tradingSkill || 0) + (backgroundBonuses.tradingSkill || 0) + (customBonuses.tradingSkill || 0),
      socialInfluence:
        (baseStats.socialInfluence || 0) +
        (backgroundBonuses.socialInfluence || 0) +
        (customBonuses.socialInfluence || 0),
    }

    return finalStats
  }

  createCharacter(name: string): MetaverseCharacter {
    if (!this.selectedClass || !this.selectedBackground) {
      throw new Error("Class and background must be selected")
    }

    const finalStats = this.calculateFinalStats()

    return {
      id: `char_${Date.now()}`,
      name,
      class: this.selectedClass,
      background: this.selectedBackground,
      stats: finalStats,
      appearance: this.appearance,
      level: 1,
      experience: 0,
      health: finalStats.constitution * 10,
      maxHealth: finalStats.constitution * 10,
      mana: finalStats.intelligence * 5,
      maxMana: finalStats.intelligence * 5,
      reputation: [],
      relationships: [...this.selectedBackground.relationships],
      inventory: [],
      web3Assets: [],
      activeQuests: [],
      completedQuests: [],
      achievements: [],
      location: "nexus-city",
      lastActive: Date.now(),
    }
  }
}

export interface CharacterAppearance {
  skinTone: string
  hairColor: string
  eyeColor: string
  height: string
  build: string
  markings: string[]
  clothing: string
}

export interface MetaverseCharacter {
  id: string
  name: string
  class: CharacterClass
  background: CharacterBackground
  stats: CharacterStats
  appearance: CharacterAppearance
  level: number
  experience: number
  health: number
  maxHealth: number
  mana: number
  maxMana: number
  reputation: MetaverseReputation[]
  relationships: NPCRelationship[]
  inventory: any[]
  web3Assets: Web3Asset[]
  activeQuests: string[]
  completedQuests: string[]
  achievements: string[]
  location: string
  lastActive: number
}

export const characterCreation = new CharacterCreationSystem()
