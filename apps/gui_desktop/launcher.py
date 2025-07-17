import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from app_gui import EmulatorGUI
from config import GUI_THEME, APP_VERSION, DEVELOPER_MODE

# Setup logging
logging.basicConfig(level=logging.DEBUG if DEVELOPER_MODE else logging.INFO)
logger = logging.getLogger("Launcher")


def apply_theme(app: QApplication):
    """
    Apply light or dark theme to the application.
    """
    if GUI_THEME == "dark":
        app.setStyle("Fusion")
        dark_palette = app.palette()
        dark_palette.setColor(app.palette().Window, "#2b2b2b")
        dark_palette.setColor(app.palette().WindowText, "#f0f0f0")
        dark_palette.setColor(app.palette().Base, "#3c3f41")
        dark_palette.setColor(app.palette().Text, "#ffffff")
        app.setPalette(dark_palette)
    elif GUI_THEME == "light":
        app.setStyle("Fusion")  # Or default style
    else:
        logger.warning(f"Unknown GUI_THEME '{GUI_THEME}', using default.")


def main():
    logger.info(f"Launching Sentenial X Emulator Desktop v{APP_VERSION}")

    app = QApplication(sys.argv)
    app.setApplicationName("Sentenial X Emulator")
    app.setOrganizationName("SentenialX Labs")

    # Set app icon (optional)
    # app.setWindowIcon(QIcon("resources/icon.png"))

    apply_theme(app)

    window = EmulatorGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
