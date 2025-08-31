# src/sentenial_x/plugins/war_game.py

from apps.pentest_suite.plugins.plugin_base import PluginBase
import random
import time

class WarGamePlugin(PluginBase):
    """
    WarGame plugin for Sentenial-X.
    Simulates a turn-based battle between agents and enemies.
    """

    def __init__(self, target: str, num_turns: int = 5):
        super().__init__(target)
        self.num_turns = num_turns
        self.agent_health = 100
        self.enemy_health = 100
        self.log = []

    def _agent_attack(self):
        damage = random.randint(10, 30)
        self.enemy_health -= damage
        self.log.append(f"Agent attacks for {damage} damage. Enemy health: {max(self.enemy_health,0)}")

    def _enemy_attack(self):
        damage = random.randint(5, 25)
        self.agent_health -= damage
        self.log.append(f"Enemy attacks for {damage} damage. Agent health: {max(self.agent_health,0)}")

    def run(self):
        """
        Execute the turn-based war game simulation.
        Returns a structured result dictionary.
        """
        self.log.append(f"Starting WarGame on target: {self.target}")
        for turn in range(1, self.num_turns + 1):
            self.log.append(f"--- Turn {turn} ---")
            self._agent_attack()
            if self.enemy_health <= 0:
                self.log.append("Enemy defeated!")
                break
            self._enemy_attack()
            if self.agent_health <= 0:
                self.log.append("Agent defeated!")
                break
            time.sleep(0.1)  # simulate delay

        winner = "Agent" if self.agent_health > self.enemy_health else "Enemy"
        self.log.append(f"Battle finished. Winner: {winner}")

        return {
            "plugin": "WarGamePlugin",
            "target": self.target,
            "winner": winner,
            "agent_health": max(self.agent_health, 0),
            "enemy_health": max(self.enemy_health, 0),
            "turns": turn,
            "battle_log": self.log,
            "status": "completed"
        }
