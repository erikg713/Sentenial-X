"""
Sentenial X A.I. Desktop GUI
A PySide6-based frontend for loading plugins, configuring inputs,
and viewing live output from the Sentenial X A.I. engine.
"""

import sys
import threading
from typing import List

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit,
    QTextEdit, QPushButton, QFileDialog,
    QCheckBox, QMessageBox
)

from plugin_manager import PluginManager


class Worker(QObject):
    """
    Worker to run plugin invocation in a background thread
    and emit results back to the GUI.
    """
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, mgr: PluginManager, name: str, kwargs: dict):
        super().__init__()
        self.mgr = mgr
        self.name = name
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.mgr.run(self.name, **self.kwargs)
            # ensure we have a string
            output = result if isinstance(result, str) else str(result)
            self.finished.emit(output)
        except Exception as e:
            self.error.emit(str(e))


class AppGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial X A.I. Defender")
        self.resize(900, 600)

        # Plugin manager
        self.manager = PluginManager()

        # Main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: plugins & configuration
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        left_panel.addWidget(QLabel("Available Plugins:"))
        self.plugin_list = QListWidget()
        for name, plugin in self.manager.plugins.items():
            item = QListWidgetItem(f"{name} — {plugin.description}")
            item.setData(Qt.UserRole, name)
            self.plugin_list.addItem(item)
        self.plugin_list.currentItemChanged.connect(self.on_plugin_selected)
        left_panel.addWidget(self.plugin_list)

        left_panel.addWidget(QLabel("Rule Paths (YARA):"))
        self.rules_input = QLineEdit()
        left_panel.addWidget(self.rules_input)
        btn_browse_rules = QPushButton("Browse…")
        btn_browse_rules.clicked.connect(self.browse_rules)
        left_panel.addWidget(btn_browse_rules)

        left_panel.addWidget(QLabel("Target File/Directory:"))
        self.target_input = QLineEdit()
        left_panel.addWidget(self.target_input)
        btn_browse_target = QPushButton("Browse…")
        btn_browse_target.clicked.connect(self.browse_target)
        left_panel.addWidget(btn_browse_target)

        self.json_checkbox = QCheckBox("Output JSON")
        left_panel.addWidget(self.json_checkbox)

        btn_run = QPushButton("Run Plugin")
        btn_run.clicked.connect(self.run_plugin)
        left_panel.addWidget(btn_run)

        left_panel.addStretch()

        # Right panel: output console
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 2)
        right_panel.addWidget(QLabel("Output Console:"))
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        right_panel.addWidget(self.output_console)

    def on_plugin_selected(self):
        """
        Auto-fill placeholders or hints based on plugin.
        """
        item = self.plugin_list.currentItem()
        if not item:
            return
        name = item.data(Qt.UserRole)
        if name == "yara_scan":
            self.rules_input.setPlaceholderText("/path/to/rules1.yar;/extra/rules/")
            self.target_input.setPlaceholderText("/path/to/file_or_directory")
        else:
            # other plugins might not use rules
            self.rules_input.setPlaceholderText("(not used)")
            self.target_input.setPlaceholderText("/path/to/input")

    def browse_rules(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YARA file or directory")
        if path:
            self.rules_input.setText(path)

    def browse_target(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not path:
            # try file
            file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
            path = file_path
        if path:
            self.target_input.setText(path)

    def run_plugin(self):
        item = self.plugin_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Plugin Selected", "Please select a plugin first.")
            return

        name = item.data(Qt.UserRole)
        kwargs = {}

        # gather arguments
        rules = self.rules_input.text().strip()
        if rules:
            # support semicolon separated
            kwargs["rule_paths"] = rules.split(";")

        target = self.target_input.text().strip()
        if target:
            kwargs["target"] = target

        if self.json_checkbox.isChecked():
            kwargs["json_out"] = True

        # clear console and spawn worker thread
        self.output_console.clear()
        self.output_console.append(f">>> Running plugin '{name}'…\n")

        worker = Worker(self.manager, name, kwargs)
        worker.finished.connect(self.on_worker_finished)
        worker.error.connect(self.on_worker_error)

        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()

    def on_worker_finished(self, text: str):
        self.output_console.append(text)

    def on_worker_error(self, err: str):
        self.output_console.append(f"ERROR: {err}")

@@ def on_plugin_selected(self):
-            if typ == "select":
+            if typ == "select":
                 # unchanged…
+            elif typ == "file":
+                btn = QPushButton("Browse…")
+                lbl = QLabel("No file selected")
+                h = QHBoxLayout()
+                h.addWidget(btn); h.addWidget(lbl)
+                self.dynamic_form.addLayout(h)
+
+                def pick():
+                    path, _ = QFileDialog.getOpenFileName(self, param["label"], 
+                                                         dialog.get("start", ""), 
+                                                         dialog.get("filter", "All Files (*)"))
+                    if path:
+                        lbl.setText(path)
+                btn.clicked.connect(pick)
+                self.inputs[param["name"]] = lambda: lbl.text()
             elif typ == "int":
                 # unchanged…

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppGui()
    window.show()
    sys.exit(app.exec())
