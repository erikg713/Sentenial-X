# Import classes from sibling files to expose them at the package level
from .sandbox_env import SandboxEnv
from .network_env import NetworkTrafficEnv
from .emulator_core import EmulatorCore
from gymnasium.envs.registration import register

# Define what happens when someone does 'from envs import *'
__all__ = [
    "SandboxEnv",
    "NetworkTrafficEnv",
    "EmulatorCore",
]

# 1. Register the Malware Sandbox Emulation Environment
register(
    id='SentinelSandbox-v0',
    entry_point='envs.sandbox_env:SandboxEnv',
    max_episode_steps=1000,
)

# 2. Register a Network Traffic Analysis Environment
register(
    id='SentinelNetwork-v0',
    entry_point='envs.network_env:NetworkTrafficEnv',
    max_episode_steps=500,
)
