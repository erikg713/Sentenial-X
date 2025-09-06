// Advanced Dialogue and Social Interaction System
export interface NPC {
  id: string
  name: string
  title: string
  description: string
  personality: NPCPersonality
  location: string
  dialogues: DialogueTree
  quests: string[]
  services: NPCService[]
  relationships: Record<string, number> // playerId -> relationship level
  lore: string
  appearance: NPCAppearance
}

export interface NPCPersonality {
  traits: string[]
  likes: string[]
  dislikes: string[]
  motivations: string[]
  fears: string[]
}

export interface NPCAppearance {
  race: string
  gender: string
  age: string
  description: string
  clothing: string
  accessories: string[]
}

export interface DialogueTree {
  greeting: DialogueNode
  nodes: Record<string, DialogueNode>
}

export interface DialogueNode {
  id: string
  text: string
  speaker: "npc" | "player"
  conditions?: DialogueCondition[]
  options: any[]
  consequences?: any[]
  emotion?: "happy" | "sad" | "angry" | "neutral" | "excited" | "worried"
}

export interface DialogueCondition {
  type: "stat" | "item" | "quest" | "relationship" | "reputation" | "class" | "background"
  target: string
  operator: ">" | "<" | "=" | ">=" | "<=" | "!="
  value: any
}

export interface DialogueOption {
  id: string
  text: string
  requirements?: any
  consequences?: any[]
}

export interface DialogueConsequence {
  type: "relationship" | "pi" | "stat"
  target: string
  value: number
  message?: string
}

export interface NPCService {
  id: string
  name: string
  type: "shop" | "training" | "crafting" | "information" | "transport" | "guild"
  cost: number
  requirements?: DialogueCondition[]
  description: string
}

const NPCS: NPC[] = [
  {
    id: "nexus-mayor",
    name: "Mayor Satoshi",
    title: "Mayor of Nexus City",
    description: "The wise leader of the central hub of the Pi metaverse",
    personality: {
      traits: ["wise", "diplomatic", "visionary"],
      likes: ["progress", "unity", "innovation"],
      dislikes: ["corruption", "chaos", "greed"],
      motivations: ["building a better metaverse", "protecting citizens"],
      fears: ["economic collapse", "social unrest"],
    },
    location: "nexus-city",
    dialogues: {
      greeting: {
        id: "mayor-greeting",
        text: "Welcome to Nexus City, traveler. I am Mayor Satoshi. How may I assist you in your journey through our metaverse?",
        speaker: "npc",
        emotion: "neutral",
        options: [
          {
            id: "ask-about-city",
            text: "Tell me about Nexus City",
            requirements: {},
            consequences: [
              {
                type: "relationship",
                target: "nexus-mayor",
                value: 5,
                message: "The Mayor appreciates your interest in the city",
              },
            ],
          },
          {
            id: "ask-about-quests",
            text: "Do you have any work for me?",
            requirements: {},
            consequences: [],
          },
          {
            id: "ask-about-pi",
            text: "What can you tell me about Pi Network?",
            requirements: { class: "crypto-mage" },
            consequences: [
              {
                type: "relationship",
                target: "nexus-mayor",
                value: 10,
                message: "The Mayor is impressed by your knowledge of crypto magic",
              },
            ],
          },
        ],
      },
      nodes: {
        "city-info": {
          id: "city-info",
          text: "Nexus City is the heart of our metaverse, where all realms connect. Here you'll find traders, adventurers, and innovators from across the Pi Network. We pride ourselves on being a beacon of decentralized governance.",
          speaker: "npc",
          emotion: "happy",
          options: [
            {
              id: "ask-governance",
              text: "How does decentralized governance work here?",
              requirements: { stats: { intelligence: 12 } },
              consequences: [],
            },
            {
              id: "ask-districts",
              text: "What districts should I visit?",
              requirements: {},
              consequences: [],
            },
          ],
        },
      },
    },
    quests: ["welcome-to-nexus", "city-council-election"],
    services: [
      {
        id: "citizenship",
        name: "Grant Citizenship",
        type: "guild",
        cost: 0,
        description: "Become a citizen of Nexus City",
      },
      {
        id: "city-info",
        name: "City Information",
        type: "information",
        cost: 0,
        description: "Learn about the city and its districts",
      },
    ],
    relationships: {},
    lore: "Mayor Satoshi founded Nexus City as a neutral ground where all Pi Network participants could gather, trade, and collaborate. His vision of decentralized governance has made the city a model for other metaverse settlements.",
    appearance: {
      race: "human",
      gender: "male",
      age: "middle-aged",
      description: "A distinguished man with silver hair and kind eyes, wearing formal robes of office",
      clothing: "mayoral-robes",
      accessories: ["chain-of-office", "pi-signet-ring"],
    },
  },
  {
    id: "crystal-sage",
    name: "Lyra the Crystal Sage",
    title: "Master of Blockchain Crystals",
    description: "An ancient being who understands the deepest mysteries of the metaverse",
    personality: {
      traits: ["mysterious", "wise", "patient"],
      likes: ["knowledge", "crystals", "meditation"],
      dislikes: ["impatience", "ignorance", "violence"],
      motivations: ["preserving ancient knowledge", "teaching worthy students"],
      fears: ["knowledge being lost", "corruption of the crystals"],
    },
    location: "crystal-caves",
    dialogues: {
      greeting: {
        id: "sage-greeting",
        text: "The crystals whisper of your arrival, young one. I am Lyra, keeper of the ancient blockchain mysteries. What brings you to my sanctuary?",
        speaker: "npc",
        emotion: "neutral",
        options: [
          {
            id: "ask-crystals",
            text: "What are blockchain crystals?",
            requirements: {},
            consequences: [],
          },
          {
            id: "seek-training",
            text: "I wish to learn your ways",
            requirements: { stats: { wisdom: 14 } },
            consequences: [
              {
                type: "relationship",
                target: "crystal-sage",
                value: 15,
                message: "Lyra senses your spiritual potential",
              },
            ],
          },
          {
            id: "challenge-knowledge",
            text: "I believe I know more about the blockchain than you",
            requirements: { stats: { intelligence: 16 } },
            consequences: [
              {
                type: "relationship",
                target: "crystal-sage",
                value: -10,
                message: "Lyra is not impressed by your arrogance",
              },
            ],
          },
        ],
      },
      nodes: {},
    },
    quests: ["crystal-attunement", "ancient-knowledge"],
    services: [
      {
        id: "crystal-training",
        name: "Crystal Magic Training",
        type: "training",
        cost: 100,
        requirements: [{ type: "relationship", target: "crystal-sage", operator: ">=", value: 25 }],
        description: "Learn to harness the power of blockchain crystals",
      },
    ],
    relationships: {},
    lore: "Lyra has existed since the first blockchain was created, her consciousness intertwined with the fundamental code of decentralized networks. She guards ancient secrets that could reshape the metaverse.",
    appearance: {
      race: "ethereal",
      gender: "female",
      age: "timeless",
      description: "A luminous being with crystalline features and eyes that reflect the depths of the blockchain",
      clothing: "crystal-robes",
      accessories: ["floating-crystals", "blockchain-crown"],
    },
  },
]

export class DialogueSystem {
  private npcs: Map<string, NPC> = new Map()
  private activeDialogue: DialogueNode | null = null
  private currentNPC: NPC | null = null

  constructor() {
    NPCS.forEach((npc) => this.npcs.set(npc.id, npc))
  }

  getNPC(id: string): NPC | null {
    return this.npcs.get(id) || null
  }

  getNPCsInLocation(location: string): NPC[] {
    return Array.from(this.npcs.values()).filter((npc) => npc.location === location)
  }

  startDialogue(npcId: string, playerId: string): DialogueNode | null {
    const npc = this.npcs.get(npcId)
    if (!npc) return null

    this.currentNPC = npc
    this.activeDialogue = npc.dialogues.greeting

    // Initialize relationship if it doesn't exist
    if (!(playerId in npc.relationships)) {
      npc.relationships[playerId] = 0
    }

    return this.activeDialogue
  }

  selectDialogueOption(
    optionId: string,
    player: any,
  ): { nextNode: DialogueNode | null; consequences: DialogueConsequence[] } {
    if (!this.activeDialogue || !this.currentNPC) {
      return { nextNode: null, consequences: [] }
    }

    const option = this.activeDialogue.options.find((opt) => opt.id === optionId)
    if (!option) {
      return { nextNode: null, consequences: [] }
    }

    // Check requirements
    if (option.requirements && !this.checkRequirements(option.requirements, player)) {
      return { nextNode: null, consequences: [] }
    }

    // Apply consequences
    const consequences = option.consequences || []
    this.applyConsequences(consequences, player)

    // Find next node
    const nextNodeId = option.id // Simplified - in real system would have proper node linking
    const nextNode = this.currentNPC.dialogues.nodes[nextNodeId] || null

    this.activeDialogue = nextNode

    return { nextNode, consequences }
  }

  private checkRequirements(requirements: any, player: any): boolean {
    // Implement requirement checking logic
    if (requirements.stats) {
      for (const [stat, value] of Object.entries(requirements.stats)) {
        if (player.stats[stat] < value) return false
      }
    }

    if (requirements.class && player.class.id !== requirements.class) {
      return false
    }

    return true
  }

  private applyConsequences(consequences: DialogueConsequence[], player: any): void {
    consequences.forEach((consequence) => {
      switch (consequence.type) {
        case "relationship":
          if (this.currentNPC) {
            this.currentNPC.relationships[player.id] =
              (this.currentNPC.relationships[player.id] || 0) + consequence.value
          }
          break
        case "pi":
          player.balance += consequence.value
          break
        case "stat":
          if (player.stats[consequence.target]) {
            player.stats[consequence.target] += consequence.value
          }
          break
      }
    })
  }

  endDialogue(): void {
    this.activeDialogue = null
    this.currentNPC = null
  }

  getRelationshipLevel(npcId: string, playerId: string): number {
    const npc = this.npcs.get(npcId)
    return npc?.relationships[playerId] || 0
  }

  getRelationshipTitle(level: number): string {
    if (level >= 80) return "Best Friend"
    if (level >= 60) return "Close Friend"
    if (level >= 40) return "Friend"
    if (level >= 20) return "Acquaintance"
    if (level >= 0) return "Neutral"
    if (level >= -20) return "Dislike"
    if (level >= -40) return "Hostile"
    if (level >= -60) return "Enemy"
    return "Nemesis"
  }
}

export const dialogueSystem = new DialogueSystem()
