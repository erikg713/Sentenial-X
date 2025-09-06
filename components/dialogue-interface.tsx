"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { dialogueSystem } from "@/lib/dialogue-system"
import type { MetaverseCharacter } from "@/lib/character-system"
import { MessageCircle, User, Heart, Coins, Star } from "lucide-react"

interface DialogueInterfaceProps {
  dialogue: any
  character: MetaverseCharacter
  onClose: () => void
  onDialogueComplete: (consequences: any[]) => void
}

export function DialogueInterface({ dialogue, character, onClose, onDialogueComplete }: DialogueInterfaceProps) {
  const [currentNode, setCurrentNode] = useState(dialogue.dialogue)
  const [conversationHistory, setConversationHistory] = useState<any[]>([])

  const handleOptionSelect = (optionId: string) => {
    const result = dialogueSystem.selectDialogueOption(optionId, character)

    // Add current dialogue to history
    setConversationHistory((prev) => [
      ...prev,
      {
        speaker: "npc",
        text: currentNode.text,
        emotion: currentNode.emotion,
      },
    ])

    // Add player response to history
    const selectedOption = currentNode.options.find((opt: any) => opt.id === optionId)
    if (selectedOption) {
      setConversationHistory((prev) => [
        ...prev,
        {
          speaker: "player",
          text: selectedOption.text,
        },
      ])
    }

    if (result.nextNode) {
      setCurrentNode(result.nextNode)
    } else {
      // End dialogue
      onDialogueComplete(result.consequences)
      onClose()
    }
  }

  const npc = dialogueSystem.getNPC(dialogue.npcId)
  const relationshipLevel = dialogueSystem.getRelationshipLevel(dialogue.npcId, character.id)
  const relationshipTitle = dialogueSystem.getRelationshipTitle(relationshipLevel)

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="bg-black/90 border-white/20 text-white max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">{npc?.name}</h3>
              <p className="text-sm text-gray-300">{npc?.title}</p>
              <Badge variant="secondary" className="text-xs mt-1">
                {relationshipTitle} ({relationshipLevel})
              </Badge>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Conversation History */}
          {conversationHistory.length > 0 && (
            <div className="max-h-32 overflow-y-auto space-y-2 bg-white/5 p-3 rounded">
              {conversationHistory.map((entry, index) => (
                <div
                  key={index}
                  className={`text-sm ${entry.speaker === "player" ? "text-blue-300" : "text-gray-300"}`}
                >
                  <strong>{entry.speaker === "player" ? character.name : npc?.name}:</strong> {entry.text}
                </div>
              ))}
            </div>
          )}

          {/* Current Dialogue */}
          <Card className="bg-white/5 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <MessageCircle className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-white leading-relaxed">{currentNode.text}</p>
                  {currentNode.emotion && (
                    <Badge variant="outline" className="mt-2 text-xs border-white/20">
                      {currentNode.emotion}
                    </Badge>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Dialogue Options */}
          <div className="space-y-2">
            {currentNode.options.map((option: any) => {
              const canSelect = !option.requirements || checkRequirements(option.requirements, character)

              return (
                <Button
                  key={option.id}
                  onClick={() => handleOptionSelect(option.id)}
                  disabled={!canSelect}
                  variant="outline"
                  className={`w-full text-left justify-start p-4 h-auto border-white/20 text-white bg-transparent hover:bg-white/10 ${
                    !canSelect ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs font-bold text-white">
                        {String.fromCharCode(65 + currentNode.options.indexOf(option))}
                      </span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm leading-relaxed">{option.text}</p>
                      {option.requirements && (
                        <div className="flex items-center gap-2 mt-2">
                          {option.requirements.stats && (
                            <Badge variant="secondary" className="text-xs">
                              Requires stats
                            </Badge>
                          )}
                          {option.requirements.class && (
                            <Badge variant="secondary" className="text-xs">
                              {option.requirements.class} only
                            </Badge>
                          )}
                        </div>
                      )}
                      {option.consequences && option.consequences.length > 0 && (
                        <div className="flex items-center gap-2 mt-2">
                          {option.consequences.map((consequence: any, index: number) => (
                            <div key={index} className="flex items-center gap-1">
                              {consequence.type === "relationship" && <Heart className="w-3 h-3 text-red-500" />}
                              {consequence.type === "pi" && <Coins className="w-3 h-3 text-yellow-500" />}
                              {consequence.type === "stat" && <Star className="w-3 h-3 text-blue-500" />}
                              <span className="text-xs text-gray-400">
                                {consequence.value > 0 ? "+" : ""}
                                {consequence.value}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </Button>
              )
            })}
          </div>

          {/* End Dialogue Option */}
          <Button
            onClick={onClose}
            variant="outline"
            className="w-full border-white/20 text-white bg-transparent hover:bg-white/10"
          >
            End Conversation
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function checkRequirements(requirements: any, character: MetaverseCharacter): boolean {
  if (requirements.stats) {
    for (const [stat, value] of Object.entries(requirements.stats)) {
      if (character.stats[stat as keyof typeof character.stats] < (value as number)) {
        return false
      }
    }
  }

  if (requirements.class && character.class.id !== requirements.class) {
    return false
  }

  return true
}
