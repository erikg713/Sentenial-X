from cyberbattle.simulation import env


def create_environment():
    """
    Initializes and returns a CyberBattleSim environment.

    Returns:
        env.CyberBattleEnv: An instance of the CyberBattleSim environment.
    """
    print("[+] Launching CyberBattleSim environment...")
    # Consider passing configuration parameters here if needed
    environment = env.CyberBattleEnv()
    if not hasattr(environment, 'reset'):
        raise RuntimeError("CyberBattleEnv instance does not have a reset method. Check installation.")
    return environment
