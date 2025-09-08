from cyberbattle.simulation import env


def create_environment():
    """Initialize CyberBattleSim environment."""
    print("[+] Launching CyberBattleSim environment...")
    environment = env.CyberBattleEnv()
    return environment
