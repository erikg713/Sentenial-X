# dashboard.py
import tkinter as tk
from core.gui.sentenialx_gui import SentenialXGUI
from utils.logger import logger

def main():
    try:
        root = tk.Tk()
        app = SentenialXGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Failed to start SentenialX GUI: {e}")

if __name__ == "__main__":
    main()
