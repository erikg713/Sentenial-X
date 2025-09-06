"use client"

import { useState, useEffect } from "react"
import { CharacterCreation } from "@/components/character-creation"
import { GameInterface } from "@/components/game-interface"
import { minePISDK } from "@/lib/minepi-sdk"
import { gameEngine } from "@/lib/game-engine"
import type { MetaverseCharacter } from "@/lib/character-system"
import { toast } from "@/hooks/use-toast"

export default function PalaceOfQuests() {
  const [character, setCharacter] = useState<MetaverseCharacter | null>(null)
  const [gameUser, setGameUser] = useState<any>(null)
  const [isInitializing, setIsInitializing] = useState(true)

  useEffect(() => {
    const initializeGame = async () => {
      try {
        // Initialize MinePI SDK
        const minePIInitialized = await minePISDK.initialize()

        if (minePIInitialized) {
          // Try to authenticate user
          const user = await minePISDK.authenticateUser()
          if (user) {
            // Create game user from MinePI user
            const gameUser = gameEngine.createUser(user.username)
            setGameUser(gameUser)
          }
        }
      } catch (error) {
        console.warn("Game initialization error:", error)
      } finally {
        setIsInitializing(false)
      }
    }

    initializeGame()
  }, [])

  const handleCharacterCreated = (newCharacter: MetaverseCharacter) => {
    setCharacter(newCharacter)

    // If we don't have a game user, create one from character
    if (!gameUser) {
      const newGameUser = gameEngine.createUser(newCharacter.name)
      setGameUser(newGameUser)
    }

    toast({
      title: "Welcome to the Pi Metaverse!",
      description: `${newCharacter.name} has entered the metaverse. Your epic adventure begins now!`,
    })
  }

  if (isInitializing) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="text-center text-white">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-bold mb-2">Initializing Pi Metaverse</h2>
          <p className="text-gray-300">Connecting to the blockchain...</p>
        </div>
      </div>
    )
  }

  if (!character) {
    return <CharacterCreation onCharacterCreated={handleCharacterCreated} />
  }

  return gameUser ? <GameInterface character={character} gameUser={gameUser} /> : null
}
