"use client"

import { useState, useEffect } from "react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Input } from "@/components/ui/input"
import { toast } from "@/hooks/use-toast"
import { WorldMap } from "@/components/world-map"
import { gameEngine, type GameUser, type GameQuest, type GameItem } from "@/lib/game-engine"
import type { DiscoveryEvent } from "@/lib/world-map"
import { Compass, Trophy, User, Menu, X, Crown, Coins, Heart, Backpack, Bell, Plus, Map } from "lucide-react"

export default function PalaceOfQuests() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [gameUser, setGameUser] = useState<GameUser | null>(null)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [username, setUsername] = useState("")
  const [notifications, setNotifications] = useState(0)
  const [activeTab, setActiveTab] = useState("explore")

  const [availableQuests, setAvailableQuests] = useState<GameQuest[]>([])
  const [activeQuests, setActiveQuests] = useState<GameQuest[]>([])

  // Initialize game data
  const refreshGameData = () => {
    const user = gameEngine.getUser()
    if (user) {
      setGameUser({ ...user })
      setAvailableQuests(gameEngine.getAvailableQuests())
      setActiveQuests(gameEngine.getActiveQuests())
    }
  }

  useEffect(() => {
    refreshGameData()
  }, [])

  const handleLogin = () => {
    if (!username.trim()) {
      toast({
        title: "Username Required",
        description: "Please enter a username to start your adventure",
        variant: "destructive",
      })
      return
    }

    const user = gameEngine.createUser(username.trim())
    setGameUser(user)
    setIsLoggedIn(true)
    refreshGameData()

    toast({
      title: "Welcome to Palace of Quests!",
      description: `Hello ${user.username}! Use WASD or arrow keys to explore the world!`,
    })
  }

  const handleBattleResult = (result: any) => {
    if (gameUser && result.victory) {
      // Update user stats
      gameEngine.addExperience(result.rewards.experience)
      setGameUser((prev) => ({
        ...prev!,
        balance: prev!.balance + result.rewards.pi,
      }))
      refreshGameData()
    }
  }

  const handleDiscovery = (discovery: DiscoveryEvent) => {
    if (gameUser) {
      gameEngine.addExperience(discovery.reward.experience)
      setGameUser((prev) => ({
        ...prev!,
        balance: prev!.balance + discovery.reward.pi,
      }))
      refreshGameData()
    }
  }

  const handleRest = () => {
    gameEngine.restoreHealth()
    refreshGameData()
    toast({
      title: "Fully Rested!",
      description: "Your health and energy have been restored",
    })
  }

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case "Legendary":
        return "bg-yellow-500"
      case "Epic":
        return "bg-purple-500"
      case "Rare":
        return "bg-blue-500"
      case "Common":
        return "bg-gray-500"
      default:
        return "bg-gray-500"
    }
  }

  const handleUseItem = (item: GameItem) => {
    if (!gameUser) return

    const success = gameEngine.useItem(item.id)
    if (success) {
      refreshGameData()
      toast({
        title: "Item Used!",
        description: `You used ${item.name}`,
      })
    }
  }

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <Card className="w-full max-w-md bg-black/20 border-white/10 text-white">
          <CardHeader className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center animate-pulse mx-auto mb-4">
              <Crown className="w-8 h-8 text-white" />
            </div>
            <CardTitle className="text-2xl">Palace of Quests</CardTitle>
            <p className="text-gray-300">Enter your name to begin your real adventure</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="bg-white/10 border-white/20 text-white placeholder:text-gray-400"
              onKeyPress={(e) => e.key === "Enter" && handleLogin()}
            />
            <Button
              onClick={handleLogin}
              className="w-full bg-purple-600 hover:bg-purple-700"
              disabled={!username.trim()}
            >
              <Crown className="w-4 h-4 mr-2" />
              Start Real Adventure
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center animate-pulse">
              <Crown className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-white">PalaceOfQuests</h1>
            <Badge variant="secondary" className="bg-green-500/20 text-green-400">
              Real World
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
                <p className="font-semibold">{gameUser?.username}</p>
                <p className="text-gray-400">Level {gameUser?.level}</p>
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
                onClick={() => setActiveTab("quests")}
              >
                <Trophy className="w-5 h-5 mr-3" />
                Quests
              </Button>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("inventory")}
              >
                <Backpack className="w-5 h-5 mr-3" />
                Inventory
              </Button>
              <Button
                variant="ghost"
                className="w-full justify-start text-white hover:bg-white/10"
                onClick={() => setActiveTab("profile")}
              >
                <User className="w-5 h-5 mr-3" />
                Profile
              </Button>
            </nav>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="p-4 space-y-6 pb-24">
        {/* Player Stats */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="bg-black/20 border-white/10 text-white">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Heart className="w-6 h-6 text-red-500" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm">
                    <span>Health</span>
                    <span>
                      {gameUser?.health}/{gameUser?.maxHealth}
                    </span>
                  </div>
                  <Progress value={((gameUser?.health || 0) / (gameUser?.maxHealth || 1)) * 100} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-black/20 border-white/10 text-white">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Coins className="w-6 h-6 text-yellow-500" />
                <div className="flex-1">
                  <div className="text-lg font-bold">{gameUser?.balance.toFixed(1)} π</div>
                  <div className="text-sm text-gray-300">Pi Balance</div>
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
            <TabsTrigger value="quests" className="text-white data-[state=active]:bg-purple-600">
              <Trophy className="w-4 h-4 mr-2" />
              Quests
            </TabsTrigger>
            <TabsTrigger value="inventory" className="text-white data-[state=active]:bg-purple-600">
              <Backpack className="w-4 h-4 mr-2" />
              Items
            </TabsTrigger>
            <TabsTrigger value="profile" className="text-white data-[state=active]:bg-purple-600">
              <User className="w-4 h-4 mr-2" />
              Profile
            </TabsTrigger>
          </TabsList>

          {/* World Map Tab */}
          <TabsContent value="explore" className="space-y-4">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Interactive World Map</h3>
              <WorldMap gameUser={gameUser} onBattleResult={handleBattleResult} onDiscovery={handleDiscovery} />
            </div>
          </TabsContent>

          {/* Quests Tab */}
          <TabsContent value="quests" className="space-y-4">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Active Quests</h3>
              {activeQuests.length === 0 ? (
                <Card className="bg-black/20 border-white/10 text-white">
                  <CardContent className="p-6 text-center">
                    <Trophy className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-300">No active quests - explore the world to find adventures!</p>
                  </CardContent>
                </Card>
              ) : (
                activeQuests.map((quest) => (
                  <Card key={quest.id} className="bg-black/20 border-white/10 text-white">
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-semibold">{quest.title}</h4>
                        <Badge className={getRarityColor(quest.difficulty)}>{quest.difficulty}</Badge>
                      </div>
                      <p className="text-sm text-gray-300 mb-3">{quest.description}</p>
                      <Progress value={quest.progress} className="mt-3" />
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </TabsContent>

          {/* Inventory Tab */}
          <TabsContent value="inventory" className="space-y-4">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Inventory</h3>
              {gameUser?.inventory.length === 0 ? (
                <Card className="bg-black/20 border-white/10 text-white">
                  <CardContent className="p-6 text-center">
                    <Backpack className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-300">Your inventory is empty</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-4">
                  {gameUser?.inventory.map((item) => (
                    <Card
                      key={item.id}
                      className="bg-black/20 border-white/10 text-white hover:bg-black/30 transition-all"
                    >
                      <CardContent className="p-4">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-semibold">{item.name}</h4>
                          <div className="flex gap-2">
                            <Badge className={getRarityColor(item.rarity)}>{item.rarity}</Badge>
                            {item.equipped && <Badge className="bg-green-500">Equipped</Badge>}
                            <Badge variant="secondary">x{item.quantity}</Badge>
                          </div>
                        </div>
                        <p className="text-sm text-gray-300 mb-3">{item.description}</p>
                        {item.type === "consumable" && (
                          <Button
                            onClick={() => handleUseItem(item)}
                            className="bg-green-600 hover:bg-green-700"
                            disabled={item.quantity <= 0}
                          >
                            Use Item
                          </Button>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          {/* Profile Tab */}
          <TabsContent value="profile" className="space-y-4">
            <Card className="bg-black/20 border-white/10 text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Avatar className="w-12 h-12">
                    <AvatarFallback>{gameUser?.username.slice(0, 2).toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="text-xl">{gameUser?.username}</h3>
                    <p className="text-gray-300">Level {gameUser?.level} World Explorer</p>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400">{gameUser?.balance.toFixed(1)} π</div>
                    <div className="text-sm text-gray-300">Pi Balance</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{gameUser?.experience}</div>
                    <div className="text-sm text-gray-300">Experience</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold">Stats</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span>Strength:</span>
                      <span>{gameUser?.stats.strength}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Defense:</span>
                      <span>{gameUser?.stats.defense}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Agility:</span>
                      <span>{gameUser?.stats.agility}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Intelligence:</span>
                      <span>{gameUser?.stats.intelligence}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-black/20 backdrop-blur-md border-t border-white/10 p-4">
        <div className="flex justify-around">
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "explore" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("explore")}
          >
            <Compass className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "quests" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("quests")}
          >
            <Trophy className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "inventory" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("inventory")}
          >
            <Backpack className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={`text-white transition-all duration-300 hover:scale-110 ${activeTab === "profile" ? "bg-purple-600" : ""}`}
            onClick={() => setActiveTab("profile")}
          >
            <User className="w-5 h-5" />
          </Button>
        </div>
      </nav>
    </div>
  )
}
