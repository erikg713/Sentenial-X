"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { WorldMap } from "@/components/world-map"
import { DialogueInterface } from "@/components/dialogue-interface"
import { toast } from "@/hooks/use-toast"
import { minePISDK } from "@/lib/minepi-sdk"
import { gameEngine } from "@/lib/game-engine"
import { dialogueSystem } from "@/lib/dialogue-system"
import type { MetaverseCharacter } from "@/lib/character-system"
import type { DiscoveryEvent } from "@/lib/world-map"
import {
  Crown,
  Heart,
  Zap,
  Coins,
  Star,
  Map,
  MessageCircle,
  Backpack,
  User,
  Trophy,
  Scroll,
  Gem,
  Users,
  Home,
} from "lucide-react"

interface GameInterfaceProps {
  character: MetaverseCharacter
  gameUser: any
}

export function GameInterface({ character, gameUser }: GameInterfaceProps) {
  const [activeTab, setActiveTab] = useState("world")
  const [playerStats, setPlayerStats] = useState(gameUser)
  const [quests, setQuests] = useState<any[]>([])
  const [inventory, setInventory] = useState<any[]>([])
  const [activeDialogue, setActiveDialogue] = useState<any>(null)
  const [showCharacterSheet, setShowCharacterSheet] = useState(false)
  const [notifications, setNotifications] = useState<string[]>([])

  useEffect(() => {
    loadGameData()
  }, [])

  const loadGameData = async () => {
    try {
      // Load quests from both game engine and MinePI
      const engineQuests = gameEngine.getAvailableQuests()
      const minePIQuests = await minePISDK.getQuests()

      setQuests([...engineQuests, ...minePIQuests])

      // Load user assets
      const assets = await minePISDK.getUserAssets()
      setInventory([...playerStats.inventory, ...assets])

      // Update player stats
      const updatedUser = gameEngine.getUser()
      if (updatedUser) {
        setPlayerStats(updatedUser)
      }
    } catch (error) {
      console.warn("Error loading game data:", error)
    }
  }

  const handleBattleResult = (result: any) => {
    if (result.victory) {
      // Update player stats
      gameEngine.addExperience(result.rewards.experience)
      const updatedUser = gameEngine.getUser()
      if (updatedUser) {
        updatedUser.balance += result.rewards.pi
        setPlayerStats({ ...updatedUser })
      }

      addNotification(`Victory! Gained ${result.rewards.experience} XP and ${result.rewards.pi} π`)
    }
    loadGameData()
  }

  const handleDiscovery = (discovery: DiscoveryEvent) => {
    gameEngine.addExperience(discovery.reward.experience)
    const updatedUser = gameEngine.getUser()
    if (updatedUser) {
      updatedUser.balance += discovery.reward.pi
      setPlayerStats({ ...updatedUser })
    }

    addNotification(
      `Discovered ${discovery.name}! Gained ${discovery.reward.experience} XP and ${discovery.reward.pi} π`,
    )
  }

  const handleQuestStart = async (questId: string) => {
    const success = gameEngine.startQuest(questId) || (await minePISDK.startQuest(questId))
    if (success) {
      toast({
        title: "Quest Started!",
        description: "Your new adventure begins!",
      })
      loadGameData()
    }
  }

  const handleQuestComplete = async (questId: string) => {
    const success = gameEngine.completeQuest(questId)
    if (success) {
      const result = await minePISDK.completeQuest(questId)
      if (result.success) {
        toast({
          title: "Quest Completed!",
          description: `Earned ${result.reward} π!`,
        })
        loadGameData()
      }
    }
  }

  const handleNPCInteraction = (npcId: string) => {
    const dialogue = dialogueSystem.startDialogue(npcId, character.id)
    if (dialogue) {
      setActiveDialogue({ npcId, dialogue })
    }
  }

  const addNotification = (message: string) => {
    setNotifications((prev) => [...prev.slice(-4), message])
    setTimeout(() => {
      setNotifications((prev) => prev.slice(1))
    }, 5000)
  }

  const getHealthPercentage = () => {
    return (playerStats.health / playerStats.maxHealth) * 100
  }

  const getEnergyPercentage = () => {
    return (playerStats.energy / playerStats.maxEnergy) * 100
  }

  const getExperiencePercentage = () => {
    const requiredExp = playerStats.level * 100
    return (playerStats.experience / requiredExp) * 100
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header with Character Info */}
        <Card className="bg-black/20 border-white/10 text-white">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
                  <Crown className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold">{character.name}</h1>
                  <p className="text-gray-300">
                    {character.class.name} • Level {playerStats.level}
                  </p>
                  <div className="flex items-center gap-4 mt-2">
                    <div className="flex items-center gap-2">
                      <Coins className="w-4 h-4 text-yellow-500" />
                      <span>{playerStats.balance.toFixed(2)} π</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Star className="w-4 h-4 text-blue-500" />
                      <span>{playerStats.experience} XP</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="text-right space-y-2">
                <div className="flex items-center gap-2">
                  <Heart className="w-4 h-4 text-red-500" />
                  <div className="w-32">
                    <Progress value={getHealthPercentage()} className="h-2" />
                  </div>
                  <span className="text-sm">
                    {playerStats.health}/{playerStats.maxHealth}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-blue-500" />
                  <div className="w-32">
                    <Progress value={getEnergyPercentage()} className="h-2" />
                  </div>
                  <span className="text-sm">
                    {playerStats.energy}/{playerStats.maxEnergy}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Star className="w-4 h-4 text-yellow-500" />
                  <div className="w-32">
                    <Progress value={getExperiencePercentage()} className="h-2" />
                  </div>
                  <span className="text-sm">Level {playerStats.level}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        {notifications.length > 0 && (
          <div className="fixed top-4 right-4 space-y-2 z-50">
            {notifications.map((notification, index) => (
              <Card key={index} className="bg-green-600/90 border-green-500 text-white animate-in slide-in-from-right">
                <CardContent className="p-3">
                  <p className="text-sm">{notification}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Main Game Interface */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-6 bg-black/20 border-white/10">
            <TabsTrigger value="world" className="flex items-center gap-2">
              <Map className="w-4 h-4" />
              World
            </TabsTrigger>
            <TabsTrigger value="quests" className="flex items-center gap-2">
              <Scroll className="w-4 h-4" />
              Quests
            </TabsTrigger>
            <TabsTrigger value="inventory" className="flex items-center gap-2">
              <Backpack className="w-4 h-4" />
              Inventory
            </TabsTrigger>
            <TabsTrigger value="social" className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              Social
            </TabsTrigger>
            <TabsTrigger value="character" className="flex items-center gap-2">
              <User className="w-4 h-4" />
              Character
            </TabsTrigger>
            <TabsTrigger value="guild" className="flex items-center gap-2">
              <Crown className="w-4 h-4" />
              Guild
            </TabsTrigger>
          </TabsList>

          {/* World Map Tab */}
          <TabsContent value="world" className="space-y-4">
            <WorldMap gameUser={playerStats} onBattleResult={handleBattleResult} onDiscovery={handleDiscovery} />
          </TabsContent>

          {/* Quests Tab */}
          <TabsContent value="quests" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Scroll className="w-5 h-5 text-yellow-500" />
                    Available Quests
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {quests
                    .filter((q) => q.status === "available")
                    .map((quest) => (
                      <Card key={quest.id} className="bg-white/5 border-white/10">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="font-semibold text-white">{quest.title}</h4>
                              <p className="text-sm text-gray-300 mt-1">{quest.description}</p>
                              <div className="flex items-center gap-2 mt-2">
                                <Badge variant={quest.difficulty === "Legendary" ? "default" : "secondary"}>
                                  {quest.difficulty}
                                </Badge>
                                <span className="text-sm text-yellow-500">{quest.reward?.pi || quest.reward} π</span>
                              </div>
                            </div>
                            <Button
                              size="sm"
                              onClick={() => handleQuestStart(quest.id)}
                              className="bg-green-600 hover:bg-green-700"
                            >
                              Start
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </CardContent>
              </Card>

              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="w-5 h-5 text-blue-500" />
                    Active Quests
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {quests
                    .filter((q) => q.status === "active")
                    .map((quest) => (
                      <Card key={quest.id} className="bg-white/5 border-white/10">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="font-semibold text-white">{quest.title}</h4>
                              <div className="mt-2">
                                <Progress value={quest.progress || 0} className="h-2" />
                                <p className="text-xs text-gray-400 mt-1">{quest.progress || 0}% Complete</p>
                              </div>
                            </div>
                            <Button
                              size="sm"
                              onClick={() => handleQuestComplete(quest.id)}
                              disabled={(quest.progress || 0) < 100}
                              className="bg-blue-600 hover:bg-blue-700"
                            >
                              Complete
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Inventory Tab */}
          <TabsContent value="inventory" className="space-y-4">
            <Card className="bg-black/20 border-white/10 text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Backpack className="w-5 h-5 text-green-500" />
                  Inventory & Assets
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {inventory.map((item, index) => (
                    <Card
                      key={`${item.id}-${index}`}
                      className="bg-white/5 border-white/10 hover:bg-white/10 transition-colors"
                    >
                      <CardContent className="p-3 text-center">
                        <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg mx-auto mb-2 flex items-center justify-center">
                          <Gem className="w-6 h-6 text-white" />
                        </div>
                        <h4 className="text-sm font-semibold text-white truncate">{item.name}</h4>
                        <Badge variant="secondary" className="text-xs mt-1">
                          {item.rarity}
                        </Badge>
                        {item.quantity && <p className="text-xs text-gray-400 mt-1">x{item.quantity}</p>}
                        {item.equipped && <Badge className="text-xs mt-1 bg-green-600">Equipped</Badge>}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Social Tab */}
          <TabsContent value="social" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MessageCircle className="w-5 h-5 text-blue-500" />
                    NPCs & Dialogue
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button
                    onClick={() => handleNPCInteraction("nexus-mayor")}
                    className="w-full justify-start bg-white/5 hover:bg-white/10 border-white/10"
                    variant="outline"
                  >
                    <User className="w-4 h-4 mr-2" />
                    Talk to Mayor Satoshi
                  </Button>
                  <Button
                    onClick={() => handleNPCInteraction("crystal-sage")}
                    className="w-full justify-start bg-white/5 hover:bg-white/10 border-white/10"
                    variant="outline"
                  >
                    <Star className="w-4 h-4 mr-2" />
                    Visit Crystal Sage
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-green-500" />
                    Relationships
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {character.relationships.map((rel) => (
                    <div key={rel.npcId} className="flex items-center justify-between">
                      <span className="text-sm">{rel.npcId.replace("-", " ")}</span>
                      <Badge variant={rel.level > 0 ? "default" : "destructive"}>
                        {rel.relationship} ({rel.level})
                      </Badge>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Character Tab */}
          <TabsContent value="character" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <User className="w-5 h-5 text-purple-500" />
                    Character Stats
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Strength:</span>
                        <span className="font-bold">{character.stats.strength}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dexterity:</span>
                        <span className="font-bold">{character.stats.dexterity}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Intelligence:</span>
                        <span className="font-bold">{character.stats.intelligence}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Wisdom:</span>
                        <span className="font-bold">{character.stats.wisdom}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Constitution:</span>
                        <span className="font-bold">{character.stats.constitution}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Charisma:</span>
                        <span className="font-bold">{character.stats.charisma}</span>
                      </div>
                    </div>
                  </div>
                  <div className="border-t border-white/10 pt-4">
                    <h4 className="font-semibold mb-2">Web3 Stats</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Mining Power:</span>
                        <span className="font-bold">{character.stats.miningPower}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Trading Skill:</span>
                        <span className="font-bold">{character.stats.tradingSkill}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Social Influence:</span>
                        <span className="font-bold">{character.stats.socialInfluence}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-black/20 border-white/10 text-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Crown className="w-5 h-5 text-yellow-500" />
                    Class & Background
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-yellow-400">{character.class.name}</h4>
                    <p className="text-sm text-gray-300 mt-1">{character.class.description}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-blue-400">{character.background.name}</h4>
                    <p className="text-sm text-gray-300 mt-1">{character.background.description}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Abilities</h4>
                    <div className="space-y-2">
                      {character.class.abilities.map((ability) => (
                        <div key={ability.id} className="bg-white/5 p-2 rounded">
                          <h5 className="font-semibold text-sm">{ability.name}</h5>
                          <p className="text-xs text-gray-400">{ability.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Guild Tab */}
          <TabsContent value="guild" className="space-y-4">
            <Card className="bg-black/20 border-white/10 text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Crown className="w-5 h-5 text-purple-500" />
                  Guild System
                </CardTitle>
              </CardHeader>
              <CardContent className="text-center py-8">
                <Home className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-2">No Guild</h3>
                <p className="text-gray-300 mb-4">Join or create a guild to unlock collaborative features!</p>
                <div className="space-y-2">
                  <Button className="w-full bg-purple-600 hover:bg-purple-700">Create Guild</Button>
                  <Button
                    variant="outline"
                    className="w-full border-white/20 text-white bg-transparent hover:bg-white/10"
                  >
                    Browse Guilds
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Dialogue Interface */}
        {activeDialogue && (
          <DialogueInterface
            dialogue={activeDialogue}
            character={character}
            onClose={() => setActiveDialogue(null)}
            onDialogueComplete={(consequences) => {
              addNotification("Dialogue completed!")
              loadGameData()
            }}
          />
        )}
      </div>
    </div>
  )
}
