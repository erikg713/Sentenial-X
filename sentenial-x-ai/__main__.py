import sys
import argparse
import logging

# Assuming the sentenial-x-ai directory is treated as a package
# The commands module is available via relative import
try:
    # Import the commands package from the current directory context
    from . import commands
    
    # Set up a logger for the main application. 
    # The file handler is already configured in commands/__init__.py
    logger = logging.getLogger('sentenial-x-ai')
    logger.addHandler(logging.StreamHandler(sys.stdout)) # Also output to console
    logger.setLevel(logging.INFO)
except ImportError:
    # Fallback for direct script execution or unexpected path setup
    print("Error: Could not import 'commands' module. Ensure the directory structure is correct for package execution.")
    sys.exit(1)


def parse_args():
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentenial X AI Command Line Interface.",
        epilog="Use 'help' as a command name to see available commands."
    )
    
    # Arguments for the command itself
    parser.add_argument(
        'command_name', 
        type=str, 
        help='The name of the command to execute (e.g., "hello", "analyze").'
    )
    parser.add_argument(
        'args', 
        nargs=argparse.REMAINDER, 
        help='Arguments passed to the command function.'
    )
    return parser.parse_args()


def main():
    """Main function to load commands and execute the requested action."""
    
    # 1. Load all commands defined in the 'commands' package
    logger.info("Initializing Sentenial X AI...")
    commands.load_commands_from_package()
    logger.info("Command modules loaded successfully.")
    
    available_commands = commands.get_available_commands()
    
    if not available_commands:
        logger.warning("No commands were registered. Check command modules for '@register_command' usage.")

    try:
        args = parse_args()
    except Exception as e:
        # Catch unexpected parsing errors
        logger.error(f"Argument parsing error: {e}")
        return

    command_name = args.command_name.lower()
    
    if command_name in ('help', 'commands'):
        print("\nAvailable Commands:")
        if available_commands:
            # Determine max command name length for neat alignment
            max_len = max(len(cmd) for cmd in available_commands.keys()) if available_commands else 0
            for name, desc in available_commands.items():
                print(f"  {name:<{max_len}} : {desc}")
        else:
            print("  (None registered)")
        print("\nUsage: python -m sentenial-x-ai <command_name> [args...]")
        return
    
    try:
        # 2. Execute the requested command
        result = commands.execute_command(command_name, *args.args)
        
        # 3. Display the result
        if result is not None:
            logger.info(f"Command '{command_name}' finished. Result:\n{result}")
            
    except commands.UnknownCommandError as e:
        logger.error(str(e))
        print(f"\nCommand '{command_name}' not found. Use 'help' to see available commands.")
        sys.exit(1)
    except commands.CommandExecutionError as e:
        logger.error(str(e))
        print(f"\nExecution failed for command '{command_name}'. See logs for details.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
