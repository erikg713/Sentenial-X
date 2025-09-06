"use client"

import { useState, useEffect } from "react"
import { CharacterCreation } from "@/components/character-creation"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { toast } from "@/hooks/use-toast"
import { WorldMap } from "@/components/world-map"
import { dialogueSystem, type NPC } from "@/lib/dialogue-system"
import type { MetaverseCharacter } from "@/lib/character-system"
import type { DiscoveryEvent } from "@/lib/world-map"
import {
  Crown,
  Users,
  MessageCircle,
  Coins,
  Heart,
  Zap,
  Map,
  Trophy,
  Backpack,
  User,
  Menu,
  X,
  Bell,
  Plus,
  Sparkles,
  Globe,
  Shield,
} from "lucide-react"

export default function PalaceOfQuests() {
  const [character, setCharacter] = useState<MetaverseCharacter | null>(null)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [notifications, setNotifications] = useState(0)
  const [activeTab, setActiveTab] = useState("explore")
  const [selectedNPC, setSelectedNPC] = useState<NPC | null>(null)
  const [dialogueNode, setDialogueNode] = useState<any>(null)
  const [nearbyNPCs, setNearbyNPCs] = useState<NPC[]>([])

  useEffect(() => {
    if (character) {
      // Load NPCs in current location
      const npcs = dialogueSystem.getNPCsInLocation(character.location)
      setNearbyNPCs(npcs)
    }
  }, [character]) // Updated to use character instead of character?.location

  const handleCharacterCreated = (newCharacter: MetaverseCharacter) => {
    setCharacter(newCharacter)
    toast({
      title: "Welcome to the Metaverse!",
      description: `${newCharacter.name} has entered the Pi Network metaverse. Your adventure begins now!`,
    })
  }

  const handleBattleResult = (result: any) => {
    if (character && result.victory) {
      // Update character with battle rewards
      setCharacter((prev) => ({
        ...prev!,
        experience: prev!.experience + result.rewards.experience,
        // Add Pi rewards to web3 assets
        web3Assets: [
          ...prev!.web3Assets,
          {
            id: `pi-reward-${Date.now()}`,
            name: "Battle Pi Tokens",
            type: "token" as const,
            rarity: "Common" as const,
            attributes: { amount: result.rewards.pi },
            owner: prev!.id,
            tradeable: true,
            stakeable: true,
            utility: ["currency", "staking"],
            metaverseValue: result.rewards.pi,
          },
        ],
      }))
    }
  }

  const handleDiscovery = (discovery: DiscoveryEvent) => {
    if (character) {
      setCharacter((prev) => ({
        ...prev!,
        experience: prev!.experience + discovery.reward.experience,
        web3Assets: [
          ...prev!.web3Assets,
          {
            id: `discovery-${Date.now()}`,
            name: `Discovery Reward: ${discovery.name}`,
            type: "token" as const,
            rarity: "Rare" as const,
            attributes: {
              experience: discovery.reward.experience,
              pi: discovery.reward.pi,
              location: discovery.name,
            },
            owner: prev!.id,
            tradeable: true,
            stakeable: false,
            utility: ["achievement", "reputation"],
            metaverseValue: discovery.reward.pi,
          },
        ],
      }))
    }
  }

  const handleNPCInteraction = (npc: NPC) => {
    if (!character) return

    setSelectedNPC(npc)
    const dialogue = dialogueSystem.startDialogue(npc.id, character.id)
    setDialogueNode(dialogue)
  }

  const handleDialogueOption = (optionId: string) => {
    if (!character || !selectedNPC) return

    const result = dialogueSystem.selectDialogueOption(optionId, character)

    if (result.consequences.length > 0) {
      result.consequences.forEach((consequence) => {
        toast({
          title: "Social Interaction",
          description: consequence.message,
        })
      })
    }

    setDialogueNode(result.nextNode)

    if (!result.nextNode) {
      setSelectedNPC(null)
      dialogueSystem.endDialogue()
    }
  }

  const handleRest = () => {
    if (!character) return

    setCharacter((prev) => ({
      ...prev!,
      health: prev!.maxHealth,
      mana: prev!.maxMana,
    }))

    toast({
      title: "Fully Rested!",
      description: "Your health and mana have been restored",
    })
  }

  const getClassIcon = (classId: string) => {
    switch (classId) {
      case "paladin":
        return <Crown className="w-6 h-6 text-yellow-500" />
      case "crypto-mage":
        return <Sparkles className="w-6 h-6 text-purple-500" />
      case "nft-hunter":
        return <Shield className="w-6 h-6 text-green-500" />
      case "defi-merchant":
        return <Coins className="w-6 h-6 text-blue-500" />
      default:
        return <User className="w-6 h-6 text-gray-500" />
    }
  }

  if (!character) {
    return <CharacterCreation onCharacterCreated={handleCharacterCreated} />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center animate-pulse">
              <Globe className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Pi Metaverse</h1>
              <p className="text-xs text-gray-300">
                {character.name} • Level {character.level}
              </p>
            </div>
            <Badge variant="secondary" className="bg-green-500/20 text-green-400">
              Web3 RPG
            </Badge>
          </div>

          <div className="flex items-center gap-3">
            {/* Rest Button */}
            <Button variant="ghost" size="icon" className="text-white" onClick={handleRest}>
              <Plus className="w-5 h-5" />
            </Button>

            {/* Notifications */}
            <Button variant="ghost" size="icon" className="text-white relative">
              <Bell className="w-5 h-5" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center animate-bounce">
                  {notifications}
                </span>
              )}
            </Button>

            <Button variant="ghost" size="icon" className="text-white" onClick={() => setIsMenuOpen(!isMenuOpen)}>
              {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      {isMenuOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm">
          <div className="absolute right-0 top-16 w-64 h-full bg-black/90 backdrop-blur-md p-4 animate-in slide-in-from-right">
            <nav className="space-y-4">
              <div className="text-white text-sm mb-4">
                <div className="flex items-center gap-2 mb-2">
                  {getClassIcon(character.class.id)}
                  <div>
                    <p className="font-semibold">{character.name}</p>
                    <p className="text-gray-400">{character.class.name}</p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">{character.background.name}</p>
              </div>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("explore")}
              >
                <Map className="w-5 h-5 mr-3" />
                World Map
              </Button>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("social")}
              >
                <Users className="w-5 h-5 mr-3" />
                Social Hub
              </Button>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("assets")}
              >
                <Backpack className="w-5 h-5 mr-3" />
                Web3 Assets
              </Button>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("character")}
              >
                <User className="w-5 h-5 mr-3" />
                Character
              </Button>
            </nav>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="p-4 space-y-6 pb-24">
        {/* Character Status */}
        <div className="grid grid-cols-3 gap-4">
          <Card className="bg-black/20 border-white/10 text-white">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Heart className="w-6 h-6 text-red-500" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm">
                    <span>Health</span>
                    <span>
                      {character.health}/{character.maxHealth}
                    </span>
                  </div>
                  <Progress value={(character.health / character.maxHealth) * 100} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-black/20 border-white/10 text-white">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Zap className="w-6 h-6 text-blue-500" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm">
                    <span>Mana</span>
                    <span>
                      {character.mana}/{character.maxMana}
                    </span>
                  </div>
                  <Progress value={(character.mana / character.maxMana) * 100} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-black/20 border-white/10 text-white">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Trophy className="w-6 h-6 text-yellow-500" />
                <div className="flex-1">
                  <div className="text-lg font-bold">{character.level}</div>
                  <div className="text-sm text-gray-300">Level</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs Navigation */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-black/20 border-white/10">
            <TabsTrigger value="explore" className="text-white data-[state=active]:bg-purple-600">
              <Map className="w-4 h-4 mr-2" />
              Explore
            </TabsTrigger>
            <TabsTrigger value="social" className="text-white data-[state=active]:bg-purple-600">
              <Users className="w-4 h-4 mr-2" />
              Social
            </TabsTrigger>
            <TabsTrigger value="assets" className="text-white data-[state=active]:bg-purple-600">
              <Backpack className="w-4 h-4 mr-2" />
              Assets
            </TabsTrigger>
            <TabsTrigger value="character" className="text-white data-[state=active]:bg-purple-600">
              <User className="w-4 h-4 mr-2" />
              Character
            </TabsTrigger>
          </TabsList>

          {/* World Map Tab */}
          <TabsContent value="explore" className="space-y-4">
            <WorldMap gameUser={character} onBattleResult={handleBattleResult} onDiscovery={handleDiscovery} />
          </TabsContent>

          {/* Social Hub Tab */}
          <TabsContent value="social" className="space-y-4">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Social Hub - NPCs Nearby</h3>
              {nearbyNPCs.length === 0 ? (
                <Card className="bg-black/20 border-white/10 text-white">
                  <CardContent className="p-6 text-center">
                    <Users className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-300">
                      No NPCs in this location. Explore to find characters to interact with!
                    </p>
                  </CardContent>
                </Card>
              ) : (
                nearbyNPCs.map((npc) => (
                  <Card
                    key={npc.id}
                    className="bg-black/20 border-white/10 text-white hover:bg-black/30 transition-all cursor-pointer"
                    onClick={() => handleNPCInteraction(npc)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center gap-4">
                        <Avatar className="w-12 h-12">
                          <AvatarFallback>{npc.name.slice(0, 2)}</AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-semibold">{npc.name}</h4>
                            <Badge variant="secondary" className="text-xs">
                              {npc.title}
                            </Badge>
                          </div>
                          <p className="text-sm text-gray-300 mb-2">{npc.description}</p>
                          <div className="flex items-center gap-4 text-xs text-gray-400">
                            <span>
                              Relationship:{" "}
                              {dialogueSystem.getRelationshipTitle(
                                dialogueSystem.getRelationshipLevel(npc.id, character.id),
                              )}
                            </span>
                            <span>Services: {npc.services.length}</span>
                          </div>
                        </div>
                        <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                          <MessageCircle className="w-4 h-4 mr-2" />
                          Talk
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </TabsContent>

          {/* Web3 Assets Tab */}
          <TabsContent value="assets" className="space-y-4">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Web3 Assets & NFTs</h3>
              {character.web3Assets.length === 0 ? (
                <Card className="bg-black/20 border-white/10 text-white">
                  <CardContent className="p-6 text-center">
                    <Backpack className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-300">
                      No Web3 assets yet. Complete quests and battles to earn tokens and NFTs!
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-4">
                  {character.web3Assets.map((asset) => (
                    <Card key={asset.id} className="bg-black/20 border-white/10 text-white">
                      <CardContent className="p-4">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-semibold">{asset.name}</h4>
                          <div className="flex gap-2">
                            <Badge
                              className={`${
                                asset.rarity === "Mythic"
                                  ? "bg-red-500"
                                  : asset.rarity === "Legendary"
                                    ? "bg-yellow-500"
                                    : asset.rarity === "Epic"
                                      ? "bg-purple-500"
                                      : asset.rarity === "Rare"
                                        ? "bg-blue-500"
                                        : "bg-gray-500"
                              }`}
                            >
                              {asset.rarity}
                            </Badge>
                            <Badge variant="secondary">{asset.type.toUpperCase()}</Badge>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-300">Metaverse Value:</span>
                            <div className="font-bold text-yellow-400">{asset.metaverseValue} π</div>
                          </div>
                          <div>
                            <span className="text-gray-300">Tradeable:</span>
                            <div className={asset.tradeable ? "text-green-400" : "text-red-400"}>
                              {asset.tradeable ? "Yes" : "No"}
                            </div>
                          </div>
                        </div>
                        <div className="mt-3">
                          <span className="text-gray-300 text-sm">Utility:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {asset.utility.map((util) => (
                              <Badge key={util} className="text-xs bg-green-600">
                                {util}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          {/* Character Tab */}
          <TabsContent value="character" className="space-y-4">
            <Card className="bg-black/20 border-white/10 text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Avatar className="w-12 h-12">
                    <AvatarFallback>{character.name.slice(0, 2).toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="text-xl">{character.name}</h3>
                    <p className="text-gray-300">
                      {character.class.name} • {character.background.name}
                    </p>
                  </div>
                  {getClassIcon(character.class.id)}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400">
                      {character.web3Assets.reduce((sum, asset) => sum + asset.metaverseValue, 0).toFixed(1)} π
                    </div>
                    <div className="text-sm text-gray-300">Total Asset Value</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{character.experience}</div>
                    <div className="text-sm text-gray-300">Experience Points</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-3">Core Stats</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
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

                  <div>
                    <h4 className="font-semibold mb-3">Web3 Stats</h4>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="text-center">
                        <div className="text-lg font-bold text-orange-400">{character.stats.miningPower}</div>
                        <div className="text-gray-300">Mining Power</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-400">{character.stats.tradingSkill}</div>
                        <div className="text-gray-300">Trading Skill</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-purple-400">{character.stats.socialInfluence}</div>
                        <div className="text-gray-300">Social Influence</div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Class Abilities</h4>
                    <div className="space-y-2">
                      {character.class.abilities.map((ability) => (
                        <Card key={ability.id} className="bg-white/5 border-white/10">
                          <CardContent className="p-3">
                            <div className="flex justify-between items-start mb-1">
                              <h5 className="font-semibold text-sm">{ability.name}</h5>
                              <Badge variant="secondary" className="text-xs">
                                {ability.cooldown}s cooldown
                              </Badge>
                            </div>
                            <p className="text-xs text-gray-300">{ability.description}</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Relationships</h4>
                    {character.relationships.length === 0 ? (
                      <p className="text-sm text-gray-400">No relationships established yet</p>
                    ) : (
                      <div className="space-y-2">
                        {character.relationships.map((rel) => (
                          <div key={rel.npcId} className="flex justify-between items-center text-sm">
                            <span>{rel.npcId.replace("-", " ")}</span>
                            <Badge
                              className={`${
                                rel.level > 50
                                  ? "bg-green-500"
                                  : rel.level > 0
                                    ? "bg-blue-500"
                                    : rel.level > -50
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                              }`}
                            >
                              {rel.relationship} ({rel.level})
                            </Badge>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* NPC Dialogue Modal */}
      {selectedNPC && dialogueNode && (
        <Dialog
          open={!!selectedNPC}
          onOpenChange={() => {
            setSelectedNPC(null)
            setDialogueNode(null)
            dialogueSystem.endDialogue()
          }}
        >
          <DialogContent className="bg-black/90 border-white/20 text-white max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3">
                <Avatar className="w-8 h-8">
                  <AvatarFallback>{selectedNPC.name.slice(0, 2)}</AvatarFallback>
                </Avatar>
                <div>
                  <span>{selectedNPC.name}</span>
                  <p className="text-sm text-gray-400 font-normal">{selectedNPC.title}</p>
                </div>
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="bg-white/5 p-4 rounded-lg">
                <p className="text-gray-200">{dialogueNode.text}</p>
              </div>

              <div className="space-y-2">
                <h4 className="font-semibold">Response Options:</h4>
                {dialogueNode.options.map((option: any) => (
                  <Button
                    key={option.id}
                    variant="outline"
                    className="w-full text-left justify-start border-white/20 text-white bg-transparent hover:bg-white/10"
                    onClick={() => handleDialogueOption(option.id)}
                  >
                    {option.text}
                  </Button>
                ))}
              </div>

              <div className="text-xs text-gray-400">
                Relationship:{" "}
                {dialogueSystem.getRelationshipTitle(dialogueSystem.getRelationshipLevel(selectedNPC.id, character.id))}
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-black/20 backdrop-blur-md border-t border-white/10 p-4">
        <div className="flex justify-around">
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "explore" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("explore")}
          >
            <Map className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "social" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("social")}
          >
            <Users className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "assets" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("assets")}
          >
            <Backpack className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "character" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("character")}
          >
            <User className="w-5 h-5" />
          </Button>
        </div>
      </nav>
    </div>
  )
}
