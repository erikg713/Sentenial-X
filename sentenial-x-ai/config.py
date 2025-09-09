from decouple import config

# Paths / DB
DB_PATH = config('DB_PATH', default='data/agent.db')

# Intervals (seconds)
HEARTBEAT_INTERVAL = config('HEARTBEAT_INTERVAL', default=10, cast=int)
COMMAND_POLL_INTERVAL = config('COMMAND_POLL_INTERVAL', default=5, cast=int)
STRATEGY_INTERVAL = config('STRATEGY_INTERVAL', default=15, cast=int)
BROADCAST_INTERVAL = config('BROADCAST_INTERVAL', default=30, cast=int)

# Networking
NETWORK_PORT = config('NETWORK_PORT', default=8000, cast=int)
PEERS = config('PEERS', default='').split(',')

# Agent identity
AGENT_ID = config('AGENT_ID', default='sentenial-x-ai-bot')
