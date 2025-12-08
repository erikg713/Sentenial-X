"""
Sentenial X AI - Command Registry and Dispatcher

This package handles the registration, management, and execution of
commands within the Sentenial X AI system.
"""

import logging
import importlib
import pkgutil
from typing import Dict, Callable, Any, Optional, List

# Configure logging for the command package
logger = logging.getLogger(__name__)

# Registry to hold map of command_name -> handler_function
_COMMAND_REGISTRY: Dict[str, Dict[str, Any]] = {}

class CommandError(Exception):
    """Base exception for command-related errors."""
    pass

class UnknownCommandError(CommandError):
    """Raised when a requested command is not found."""
    pass

class CommandExecutionError(CommandError):
    """Raised when a command fails during execution."""
    pass

def register_command(name: str, description: str = "", aliases: Optional[List[str]] = None):
    """
    Decorator to register a function as a command.

    Args:
        name: The primary name of the command (e.g., 'analyze').
        description: A help string describing what the command does.
        aliases: A list of alternative names for the command.
    """
    def decorator(func: Callable):
        command_info = {
            "func": func,
            "description": description,
            "name": name
        }
        
        # Register primary name
        if name in _COMMAND_REGISTRY:
            logger.warning(f"Overwriting existing command registration for '{name}'")
        _COMMAND_REGISTRY[name] = command_info
        
        # Register aliases
        if aliases:
            for alias in aliases:
                _COMMAND_REGISTRY[alias] = command_info
                
        return func
    return decorator

def execute_command(command_name: str, *args, **kwargs) -> Any:
    """
    Executes a registered command by name.

    Args:
        command_name: The name or alias of the command to run.
        *args: Positional arguments to pass to the command function.
        **kwargs: Keyword arguments to pass to the command function.

    Returns:
        The result of the command function.

    Raises:
        UnknownCommandError: If the command is not found.
        CommandExecutionError: If the command function raises an exception.
    """
    if command_name not in _COMMAND_REGISTRY:
        raise UnknownCommandError(f"Command '{command_name}' is not recognized.")

    cmd_info = _COMMAND_REGISTRY[command_name]
    func = cmd_info["func"]
    
    try:
        logger.info(f"Executing command: {command_name}")
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing '{command_name}': {e}", exc_info=True)
        raise CommandExecutionError(f"Failed to execute '{command_name}': {str(e)}") from e

def get_available_commands() -> Dict[str, str]:
    """
    Returns a dictionary of unique command names and their descriptions.
    """
    unique_commands = {}
    for key, info in _COMMAND_REGISTRY.items():
        # Only add the primary name to the listing to avoid duplicates from aliases
        if info["name"] == key:
            unique_commands[key] = info["description"]
    return unique_commands

def load_commands_from_package(package_path: str = __name__):
    """
    Dynamically imports all modules in the current package to ensure 
    decorators run and commands are registered.
    """
    # Import the package itself to get the path
    package = importlib.import_module(package_path)
    
    if hasattr(package, "__path__"):
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package_path}.{module_name}"
            try:
                importlib.import_module(full_module_name)
                logger.debug(f"Loaded command module: {full_module_name}")
            except ImportError as e:
                logger.error(f"Failed to load command module {full_module_name}: {e}")

# Expose key components
__all__ = [
    "register_command",
    "execute_command",
    "get_available_commands",
    "load_commands_from_package",
    "CommandError",
    "UnknownCommandError"
    "train", "eval", "serve", "telemetry"
]
