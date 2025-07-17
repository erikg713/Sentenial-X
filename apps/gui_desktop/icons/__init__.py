"""
Sentenial X GUI Icons Package

This module provides access to all UI icons as QIcon instances,
ready to be used in Qt widgets, toolbars, and dialogs.
"""

from PyQt5.QtGui import QIcon
from pathlib import Path

ICON_ROOT = Path(__file__).parent.resolve()

# Example icons — update to match your files
icon_files = {
    "alert": "alert.png",
    "check": "checkmark.png",
    "warning": "warning.png",
    "lock": "lock.png",
    "decrypt": "decrypt.png",
    "encrypt": "encrypt.png",
}

icons = {
    name: QIcon(str(ICON_ROOT / filename))
    for name, filename in icon_files.items()
}

# Usage:
# from apps.gui_desktop.icons import icons
# button.setIcon(icons["alert"])

