def execute_command(cmd):
    if cmd == "shutdown":
        print("[ACTION] Shutting down...")
        exit(0)
    elif cmd.startswith("echo "):
        print(f"[ECHO] {cmd[5:]}")
    else:
        print(f"[UNKNOWN COMMAND] {cmd}")
