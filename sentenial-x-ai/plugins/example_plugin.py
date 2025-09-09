def register(agent_context):
    # Called on load: agent_context gives access to hooks
    agent_context.register_command("greet", greet)

def greet(args):
    return f"Hello, {args or 'world'}!"
