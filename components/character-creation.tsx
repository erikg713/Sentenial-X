"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  characterCreation,
  type CharacterClass,
  type CharacterBackground,
  type MetaverseCharacter,
} from "@/lib/character-system"
import { Crown, Wand2, Target, Coins, User } from "lucide-react"

interface CharacterCreationProps {
  onCharacterCreated: (character: MetaverseCharacter) => void
}

export function CharacterCreation({ onCharacterCreated }: CharacterCreationProps) {
  const [step, setStep] = useState(1)
  const [characterName, setCharacterName] = useState("")
  const [selectedClass, setSelectedClass] = useState<CharacterClass | null>(null)
  const [selectedBackground, setSelectedBackground] = useState<CharacterBackground | null>(null)
  const [appearance, setAppearance] = useState({
    skinTone: "medium",
    hairColor: "brown",
    eyeColor: "brown",
    height: "average",
    build: "athletic",
  })

  const classes = characterCreation.getAvailableClasses()
  const backgrounds = characterCreation.getAvailableBackgrounds()

  const getClassIcon = (classId: string) => {
    switch (classId) {
      case "paladin":
        return <Crown className="w-8 h-8 text-yellow-500" />
      case "crypto-mage":
        return <Wand2 className="w-8 h-8 text-purple-500" />
      case "nft-hunter":
        return <Target className="w-8 h-8 text-green-500" />
      case "defi-merchant":
        return <Coins className="w-8 h-8 text-blue-500" />
      default:
        return <User className="w-8 h-8 text-gray-500" />
    }
  }

  const handleClassSelect = (characterClass: CharacterClass) => {
    setSelectedClass(characterClass)
    characterCreation.selectClass(characterClass.id)
  }

  const handleBackgroundSelect = (background: CharacterBackground) => {
    setSelectedBackground(background)
    characterCreation.selectBackground(background.id)
  }

  const handleCreateCharacter = () => {
    if (!characterName.trim() || !selectedClass || !selectedBackground) return

    characterCreation.setAppearance(appearance)
    const character = characterCreation.createCharacter(characterName.trim())
    onCharacterCreated(character)
  }

  const canProceed = (currentStep: number) => {
    switch (currentStep) {
      case 1:
        return characterName.trim().length > 0
      case 2:
        return selectedClass !== null
      case 3:
        return selectedBackground !== null
      case 4:
        return true
      default:
        return false
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="max-w-4xl mx-auto">
        <Card className="bg-black/20 border-white/10 text-white">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl flex items-center justify-center gap-3">
              <Crown className="w-8 h-8 text-purple-500" />
              Create Your Metaverse Avatar
            </CardTitle>
            <p className="text-gray-300">Design your unique character for the Pi Network metaverse</p>
          </CardHeader>
          <CardContent>
            {/* Progress Steps */}
            <div className="flex justify-center mb-8">
              <div className="flex items-center space-x-4">
                {[1, 2, 3, 4, 5].map((stepNum) => (
                  <div key={stepNum} className="flex items-center">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        step >= stepNum ? "bg-purple-600 text-white" : "bg-gray-600 text-gray-300"
                      }`}
                    >
                      {stepNum}
                    </div>
                    {stepNum < 5 && <div className="w-8 h-0.5 bg-gray-600 mx-2" />}
                  </div>
                ))}
              </div>
            </div>

            {/* Step 1: Name */}
            {step === 1 && (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-2">Choose Your Name</h3>
                  <p className="text-gray-300">This will be your identity in the metaverse</p>
                </div>
                <div className="max-w-md mx-auto">
                  <Input
                    placeholder="Enter your character name"
                    value={characterName}
                    onChange={(e) => setCharacterName(e.target.value)}
                    className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 text-center text-xl"
                    maxLength={20}
                  />
                  <p className="text-sm text-gray-400 mt-2 text-center">{characterName.length}/20 characters</p>
                </div>
              </div>
            )}

            {/* Step 2: Class Selection */}
            {step === 2 && (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-2">Choose Your Class</h3>
                  <p className="text-gray-300">Each class has unique abilities and playstyles</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {classes.map((characterClass) => (
                    <Card
                      key={characterClass.id}
                      className={`cursor-pointer transition-all hover:scale-105 ${
                        selectedClass?.id === characterClass.id
                          ? "bg-purple-600/30 border-purple-500"
                          : "bg-black/20 border-white/10 hover:bg-black/30"
                      }`}
                      onClick={() => handleClassSelect(characterClass)}
                    >
                      <CardContent className="p-6">
                        <div className="flex items-center gap-4 mb-4">
                          {getClassIcon(characterClass.id)}
                          <div>
                            <h4 className="text-xl font-bold text-white">{characterClass.name}</h4>
                            <p className="text-sm text-gray-300">{characterClass.description}</p>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <h5 className="font-semibold text-white">Starting Stats:</h5>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            <div>STR: {characterClass.startingStats.strength}</div>
                            <div>DEX: {characterClass.startingStats.dexterity}</div>
                            <div>INT: {characterClass.startingStats.intelligence}</div>
                            <div>WIS: {characterClass.startingStats.wisdom}</div>
                            <div>CON: {characterClass.startingStats.constitution}</div>
                            <div>CHA: {characterClass.startingStats.charisma}</div>
                          </div>
                        </div>
                        <Separator className="my-3 bg-white/20" />
                        <div className="space-y-2">
                          <h5 className="font-semibold text-white">Web3 Stats:</h5>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            <div>Mining: {characterClass.startingStats.miningPower}</div>
                            <div>Trading: {characterClass.startingStats.tradingSkill}</div>
                            <div>Social: {characterClass.startingStats.socialInfluence}</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {/* Step 3: Background Selection */}
            {step === 3 && (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-2">Choose Your Background</h3>
                  <p className="text-gray-300">Your past shapes your relationships and starting bonuses</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {backgrounds.map((background) => (
                    <Card
                      key={background.id}
                      className={`cursor-pointer transition-all hover:scale-105 ${
                        selectedBackground?.id === background.id
                          ? "bg-blue-600/30 border-blue-500"
                          : "bg-black/20 border-white/10 hover:bg-black/30"
                      }`}
                      onClick={() => handleBackgroundSelect(background)}
                    >
                      <CardContent className="p-6">
                        <h4 className="text-xl font-bold text-white mb-2">{background.name}</h4>
                        <p className="text-sm text-gray-300 mb-4">{background.description}</p>
                        <div className="space-y-3">
                          <div>
                            <h5 className="font-semibold text-white">Stat Bonuses:</h5>
                            <div className="flex flex-wrap gap-2 mt-1">
                              {Object.entries(background.bonuses).map(([stat, value]) => (
                                <Badge key={stat} variant="secondary" className="text-xs">
                                  {stat}: +{value}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h5 className="font-semibold text-white">Starting Items:</h5>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {background.startingItems.map((item) => (
                                <Badge key={item} className="text-xs bg-green-600">
                                  {item.replace("-", " ")}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4: Appearance */}
            {step === 4 && (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-2">Customize Appearance</h3>
                  <p className="text-gray-300">Design how your avatar looks in the metaverse</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Skin Tone</label>
                      <div className="grid grid-cols-3 gap-2">
                        {["light", "medium", "dark"].map((tone) => (
                          <Button
                            key={tone}
                            variant={appearance.skinTone === tone ? "default" : "outline"}
                            onClick={() => setAppearance({ ...appearance, skinTone: tone })}
                            className="capitalize"
                          >
                            {tone}
                          </Button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Hair Color</label>
                      <div className="grid grid-cols-3 gap-2">
                        {["black", "brown", "blonde", "red", "white", "blue"].map((color) => (
                          <Button
                            key={color}
                            variant={appearance.hairColor === color ? "default" : "outline"}
                            onClick={() => setAppearance({ ...appearance, hairColor: color })}
                            className="capitalize"
                          >
                            {color}
                          </Button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Eye Color</label>
                      <div className="grid grid-cols-3 gap-2">
                        {["brown", "blue", "green", "hazel", "gray", "violet"].map((color) => (
                          <Button
                            key={color}
                            variant={appearance.eyeColor === color ? "default" : "outline"}
                            onClick={() => setAppearance({ ...appearance, eyeColor: color })}
                            className="capitalize"
                          >
                            {color}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Height</label>
                      <div className="grid grid-cols-3 gap-2">
                        {["short", "average", "tall"].map((height) => (
                          <Button
                            key={height}
                            variant={appearance.height === height ? "default" : "outline"}
                            onClick={() => setAppearance({ ...appearance, height })}
                            className="capitalize"
                          >
                            {height}
                          </Button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Build</label>
                      <div className="grid grid-cols-3 gap-2">
                        {["slim", "athletic", "stocky"].map((build) => (
                          <Button
                            key={build}
                            variant={appearance.build === build ? "default" : "outline"}
                            onClick={() => setAppearance({ ...appearance, build })}
                            className="capitalize"
                          >
                            {build}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 5: Final Review */}
            {step === 5 && (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-2">Character Summary</h3>
                  <p className="text-gray-300">Review your character before entering the metaverse</p>
                </div>
                <Card className="bg-black/20 border-white/10">
                  <CardContent className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-xl font-bold text-white">{characterName}</h4>
                          <p className="text-gray-300">
                            {selectedClass?.name} • {selectedBackground?.name}
                          </p>
                        </div>
                        <div>
                          <h5 className="font-semibold text-white mb-2">Final Stats:</h5>
                          {selectedClass && selectedBackground && (
                            <div className="grid grid-cols-3 gap-2 text-sm">
                              {Object.entries(characterCreation.calculateFinalStats()).map(([stat, value]) => (
                                <div key={stat} className="flex justify-between">
                                  <span className="capitalize">{stat.replace(/([A-Z])/g, " $1")}:</span>
                                  <span className="font-bold">{value}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div>
                          <h5 className="font-semibold text-white mb-2">Appearance:</h5>
                          <div className="text-sm space-y-1">
                            <div>Skin: {appearance.skinTone}</div>
                            <div>Hair: {appearance.hairColor}</div>
                            <div>Eyes: {appearance.eyeColor}</div>
                            <div>Height: {appearance.height}</div>
                            <div>Build: {appearance.build}</div>
                          </div>
                        </div>
                        <div>
                          <h5 className="font-semibold text-white mb-2">Starting Equipment:</h5>
                          <div className="flex flex-wrap gap-1">
                            {selectedClass?.equipment.map((item) => (
                              <Badge key={item} className="text-xs bg-purple-600">
                                {item.replace("-", " ")}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="flex justify-between mt-8">
              <Button
                variant="outline"
                onClick={() => setStep(Math.max(1, step - 1))}
                disabled={step === 1}
                className="border-white/20 text-white bg-transparent hover:bg-white/10"
              >
                Previous
              </Button>
              {step < 5 ? (
                <Button
                  onClick={() => setStep(step + 1)}
                  disabled={!canProceed(step)}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  Next
                </Button>
              ) : (
                <Button
                  onClick={handleCreateCharacter}
                  className="bg-green-600 hover:bg-green-700"
                  disabled={!canProceed(step)}
                >
                  <Crown className="w-4 h-4 mr-2" />
                  Enter Metaverse
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
