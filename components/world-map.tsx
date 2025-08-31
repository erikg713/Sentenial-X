"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { toast } from "@/hooks/use-toast"
import { worldMap, type WorldTile, type EncounterEvent, type DiscoveryEvent, type WorldEnemy } from "@/lib/world-map"
import {
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Swords,
  Home,
  Trees,
  Mountain,
  Waves,
  Pickaxe,
  Crown,
  Skull,
  MapPin,
  Eye,
  EyeOff,
  Zap,
  Heart,
  Shield,
  Star,
  Coins,
} from "lucide-react"

interface WorldMapProps {
  gameUser: any
  onBattleResult: (result: any) => void
  onDiscovery: (discovery: DiscoveryEvent) => void
}

export function WorldMap({ gameUser, onBattleResult, onDiscovery }: WorldMapProps) {
  const [visibleTiles, setVisibleTiles] = useState<WorldTile[]>([])
  const [playerPosition, setPlayerPosition] = useState(worldMap.getPlayerPosition())
  const [selectedTile, setSelectedTile] = useState<WorldTile | null>(null)
  const [encounterDialog, setEncounterDialog] = useState<EncounterEvent | null>(null)
  const [isMoving, setIsMoving] = useState(false)
  const [showMiniMap, setShowMiniMap] = useState(false)

  const refreshMap = useCallback(() => {
    setVisibleTiles(worldMap.getVisibleTiles())
    setPlayerPosition(worldMap.getPlayerPosition())
  }, [])

  useEffect(() => {
    refreshMap()
  }, [refreshMap])

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (isMoving) return

      switch (e.key.toLowerCase()) {
        case "w":
        case "arrowup":
          handleMove("north")
          break
        case "s":
        case "arrowdown":
          handleMove("south")
          break
        case "a":
        case "arrowleft":
          handleMove("west")
          break
        case "d":
        case "arrowright":
          handleMove("east")
          break
        case "m":
          setShowMiniMap(!showMiniMap)
          break
      }
    }

    window.addEventListener("keydown", handleKeyPress)
    return () => window.removeEventListener("keydown", handleKeyPress)
  }, [isMoving, showMiniMap])

  const handleMove = async (direction: "north" | "south" | "east" | "west") => {
    if (isMoving) return

    setIsMoving(true)

    const result = worldMap.movePlayer(direction)

    if (!result.success) {
      toast({
        title: "Cannot Move",
        description: result.message || "Path blocked!",
        variant: "destructive",
      })
      setIsMoving(false)
      return
    }

    // Update position immediately for smooth movement
    setPlayerPosition(result.newPosition)
    refreshMap()

    // Handle discoveries
    result.discoveries.forEach((discovery) => {
      onDiscovery(discovery)
      toast({
        title: "Discovery!",
        description: `You discovered ${discovery.name}! Gained ${discovery.reward.experience} XP and ${discovery.reward.pi} π`,
      })
    })

    // Handle encounters
    if (result.encounters.length > 0) {
      const encounter = result.encounters[0] // Handle first encounter
      setEncounterDialog(encounter)
    }

    // Add movement delay for realism
    setTimeout(() => {
      setIsMoving(false)
    }, 300)
  }

  const handleBattle = async (enemy: WorldEnemy) => {
    const battleResult = worldMap.battleEnemy(enemy.id)

    if (battleResult.victory) {
      toast({
        title: "Victory!",
        description: `You defeated ${enemy.name}! Gained ${battleResult.rewards.experience} XP and ${battleResult.rewards.pi} π`,
      })
      onBattleResult(battleResult)
    } else {
      toast({
        title: "Defeat!",
        description: `You were defeated by ${enemy.name}!`,
        variant: "destructive",
      })
    }

    setEncounterDialog(null)
    refreshMap()
  }

  const getTileIcon = (tile: WorldTile) => {
    switch (tile.type) {
      case "village":
        return <Home className="w-4 h-4 text-yellow-500" />
      case "forest":
        return <Trees className="w-4 h-4 text-green-500" />
      case "mountain":
        return <Mountain className="w-4 h-4 text-gray-500" />
      case "water":
        return <Waves className="w-4 h-4 text-blue-500" />
      case "cave":
        return <Pickaxe className="w-4 h-4 text-orange-500" />
      case "dungeon":
        return <Crown className="w-4 h-4 text-purple-500" />
      case "desert":
        return <div className="w-4 h-4 bg-yellow-600 rounded" />
      default:
        return <div className="w-4 h-4 bg-green-400 rounded" />
    }
  }

  const getTileColor = (tile: WorldTile) => {
    if (!tile.discovered) return "bg-black"

    switch (tile.type) {
      case "village":
        return "bg-yellow-200"
      case "forest":
        return "bg-green-300"
      case "mountain":
        return "bg-gray-400"
      case "water":
        return "bg-blue-300"
      case "cave":
        return "bg-orange-300"
      case "dungeon":
        return "bg-purple-300"
      case "desert":
        return "bg-yellow-300"
      case "swamp":
        return "bg-green-600"
      default:
        return "bg-green-200"
    }
  }

  const currentTile = worldMap.getTileAt(playerPosition.x, playerPosition.y)

  return (
    <div className="space-y-4">
      {/* Current Location Info */}
      <Card className="bg-black/20 border-white/10 text-white">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getTileIcon(currentTile!)}
              <div>
                <h3 className="font-semibold">{currentTile?.name}</h3>
                <p className="text-sm text-gray-300">{currentTile?.description}</p>
                <p className="text-xs text-gray-400">
                  Position: ({playerPosition.x}, {playerPosition.y}) • Facing: {playerPosition.facing}
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowMiniMap(!showMiniMap)}
              className="border-white/20 text-white bg-transparent hover:bg-white/10"
            >
              {showMiniMap ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              {showMiniMap ? "Hide" : "Map"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Movement Controls */}
      <Card className="bg-black/20 border-white/10 text-white">
        <CardContent className="p-4">
          <div className="text-center space-y-3">
            <p className="text-sm text-gray-300">Use WASD or arrow keys to move</p>
            <div className="grid grid-cols-3 gap-2 max-w-48 mx-auto">
              <div></div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleMove("north")}
                disabled={isMoving}
                className="border-white/20 text-white bg-transparent hover:bg-white/10"
              >
                <ArrowUp className="w-4 h-4" />
              </Button>
              <div></div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleMove("west")}
                disabled={isMoving}
                className="border-white/20 text-white bg-transparent hover:bg-white/10"
              >
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <div className="flex items-center justify-center">
                <MapPin className="w-6 h-6 text-red-500" />
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleMove("east")}
                disabled={isMoving}
                className="border-white/20 text-white bg-transparent hover:bg-white/10"
              >
                <ArrowRight className="w-4 h-4" />
              </Button>
              <div></div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleMove("south")}
                disabled={isMoving}
                className="border-white/20 text-white bg-transparent hover:bg-white/10"
              >
                <ArrowDown className="w-4 h-4" />
              </Button>
              <div></div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* World Map Grid */}
      {showMiniMap && (
        <Card className="bg-black/20 border-white/10 text-white">
          <CardContent className="p-4">
            <h3 className="font-semibold mb-3">World Map</h3>
            <div className="grid grid-cols-5 gap-1 max-w-80 mx-auto">
              {visibleTiles.map((tile) => {
                const isPlayerTile = tile.x === playerPosition.x && tile.y === playerPosition.y
                const hasEnemies = tile.enemies.some((e) => e.isAlive)
                const hasStructures = tile.structures.length > 0

                return (
                  <div
                    key={tile.id}
                    className={`
                      relative w-12 h-12 border border-white/20 rounded cursor-pointer
                      ${getTileColor(tile)} ${isPlayerTile ? "ring-2 ring-red-500" : ""}
                      ${!tile.walkable ? "opacity-50" : ""}
                      hover:ring-1 hover:ring-white/50 transition-all
                    `}
                    onClick={() => setSelectedTile(tile)}
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      {isPlayerTile ? (
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                      ) : (
                        getTileIcon(tile)
                      )}
                    </div>
                    {hasEnemies && <Skull className="absolute top-0 right-0 w-3 h-3 text-red-600" />}
                    {hasStructures && <Star className="absolute bottom-0 left-0 w-3 h-3 text-yellow-500" />}
                    {!tile.discovered && <div className="absolute inset-0 bg-black/70 rounded" />}
                  </div>
                )
              })}
            </div>
            <div className="mt-3 text-xs text-gray-400 space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full" />
                <span>Your Position</span>
                <Skull className="w-3 h-3 text-red-600 ml-4" />
                <span>Enemies</span>
                <Star className="w-3 h-3 text-yellow-500 ml-4" />
                <span>Structures</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tile Details Dialog */}
      {selectedTile && (
        <Dialog open={!!selectedTile} onOpenChange={() => setSelectedTile(null)}>
          <DialogContent className="bg-black/90 border-white/20 text-white">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {getTileIcon(selectedTile)}
                {selectedTile.name}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <p className="text-gray-300">{selectedTile.description}</p>
              <div className="text-sm space-y-2">
                <div>
                  <strong>Position:</strong> ({selectedTile.x}, {selectedTile.y})
                </div>
                <div>
                  <strong>Type:</strong> {selectedTile.type}
                </div>
                <div>
                  <strong>Walkable:</strong> {selectedTile.walkable ? "Yes" : "No"}
                </div>
              </div>

              {selectedTile.enemies.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Enemies:</h4>
                  <div className="space-y-1">
                    {selectedTile.enemies
                      .filter((e) => e.isAlive)
                      .map((enemy) => (
                        <div key={enemy.id} className="text-sm text-red-400">
                          {enemy.name} (Level {enemy.level})
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {selectedTile.structures.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Structures:</h4>
                  <div className="space-y-1">
                    {selectedTile.structures.map((structure) => (
                      <div key={structure.id} className="text-sm text-yellow-400">
                        {structure.name}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Encounter Dialog */}
      {encounterDialog && (
        <Dialog open={!!encounterDialog} onOpenChange={() => setEncounterDialog(null)}>
          <DialogContent className="bg-black/90 border-white/20 text-white">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {encounterDialog.type === "enemy" && <Swords className="w-5 h-5 text-red-500" />}
                {encounterDialog.type === "structure" && <Star className="w-5 h-5 text-yellow-500" />}
                {encounterDialog.type === "random" && <Zap className="w-5 h-5 text-purple-500" />}
                Encounter!
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              {encounterDialog.type === "enemy" && (
                <div className="space-y-4">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-red-400">{encounterDialog.data.name}</h3>
                    <Badge variant="secondary">Level {encounterDialog.data.level}</Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="text-center">
                      <Heart className="w-4 h-4 text-red-500 mx-auto mb-1" />
                      <div>{encounterDialog.data.health} HP</div>
                    </div>
                    <div className="text-center">
                      <Swords className="w-4 h-4 text-orange-500 mx-auto mb-1" />
                      <div>{encounterDialog.data.attack} ATK</div>
                    </div>
                    <div className="text-center">
                      <Shield className="w-4 h-4 text-blue-500 mx-auto mb-1" />
                      <div>{encounterDialog.data.defense} DEF</div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleBattle(encounterDialog.data)}
                      className="flex-1 bg-red-600 hover:bg-red-700"
                    >
                      <Swords className="w-4 h-4 mr-2" />
                      Fight
                    </Button>
                    <Button
                      onClick={() => setEncounterDialog(null)}
                      variant="outline"
                      className="border-white/20 text-white bg-transparent hover:bg-white/10"
                    >
                      Flee
                    </Button>
                  </div>
                </div>
              )}

              {encounterDialog.type === "structure" && (
                <div className="space-y-4">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-yellow-400">{encounterDialog.data.name}</h3>
                    <p className="text-gray-300">You found a {encounterDialog.data.type}!</p>
                  </div>
                  <div className="flex gap-2">
                    <Button className="flex-1 bg-yellow-600 hover:bg-yellow-700">
                      <Star className="w-4 h-4 mr-2" />
                      Interact
                    </Button>
                    <Button
                      onClick={() => setEncounterDialog(null)}
                      variant="outline"
                      className="border-white/20 text-white bg-transparent hover:bg-white/10"
                    >
                      Leave
                    </Button>
                  </div>
                </div>
              )}

              {encounterDialog.type === "random" && (
                <div className="space-y-4">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-purple-400">Random Event</h3>
                    <p className="text-gray-300">{encounterDialog.data.message}</p>
                  </div>
                  {encounterDialog.data.reward && (
                    <div className="flex items-center justify-center gap-4 text-sm">
                      <span className="flex items-center gap-1">
                        <Coins className="w-4 h-4 text-yellow-500" />+{encounterDialog.data.reward.pi} π
                      </span>
                      <span className="flex items-center gap-1">
                        <Star className="w-4 h-4 text-blue-500" />+{encounterDialog.data.reward.experience} XP
                      </span>
                    </div>
                  )}
                  <Button onClick={() => setEncounterDialog(null)} className="w-full bg-purple-600 hover:bg-purple-700">
                    Continue
                  </Button>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
