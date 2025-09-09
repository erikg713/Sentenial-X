import click
from sentenial_x_ai.agents.base_agent import SentenialAgent
from cyberbattle.simulation import env

@click.command("train")
@click.option("--episodes", default=1000, help="Number of training episodes.")
def train_agent(episodes: int):
    """Train a Sentenial-X RL agent in CyberBattleSim."""
    environment = env.CyberBattleEnv()
    agent = SentenialAgent(environment, episodes=episodes)
    agent.train()
    click.echo(f"[+] Training complete for {episodes} episodes.")
