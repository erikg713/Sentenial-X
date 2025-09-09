CUSTOM_COMMANDS = {}

def register_custom_command(name, func):
    CUSTOM_COMMANDS[name] = func

def execute_command(cmd):
    name, *args = cmd.split(" ", 1)
    arg = args[0] if args else ""
    if name in CUSTOM_COMMANDS:
        result = CUSTOM_COMMANDS[name](arg)
        print(f"[PLUGIN] {result}")
    elif cmd == "shutdown":
        print("[ACTION] Shutting down...")
        exit(0)
    elif cmd.startswith("echo "):
        print(f"[ECHO] {cmd[5:]}")
    else:
        print(f"[UNKNOWN COMMAND] {cmd}")
