"""
Sentenial X A.I. Desktop GUI
A PySide6-based frontend for loading plugins, configuring inputs,
and viewing live output from the Sentenial X A.I. engine.
"""

import sys
import threading
from typing import Any, Dict

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QMessageBox
)

from plugin_manager import PluginManager


class Worker(QObject):
    """
    Runs plugin invocation in a background thread and emits results to GUI.
    """
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, mgr: PluginManager, name: str, kwargs: Dict[str, Any]):
        super().__init__()
        self.mgr = mgr
        self.name = name
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.mgr.run(self.name, **self.kwargs)
            out = result if isinstance(result, str) else str(result)
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


class AppGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial X A.I. Defender")
        self.resize(900, 600)

        self.manager = PluginManager()
        self.inputs: Dict[str, Any] = {}

        # Layouts
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: plugin list + dynamic form
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

        # Dynamic form area
        self.dynamic_form = QVBoxLayout()
        left_panel.addLayout(self.dynamic_form)

        # JSON toggle and run
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
        """Rebuild inputs based on selected plugin.parameters."""
        # Clear old widgets
        while self.dynamic_form.count():
            item = self.dynamic_form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.inputs.clear()

        item = self.plugin_list.currentItem()
        if not item:
            return
        name = item.data(Qt.UserRole)
        plugin = self.manager.plugins[name]

        for param in getattr(plugin, "parameters", []):
            lbl = QLabel(param["label"] + ":")
            self.dynamic_form.addWidget(lbl)
            ptype = param["type"]

            if ptype == "select":
                combo = QComboBox()
                for choice in param.get("choices", []):
                    combo.addItem(choice)
                if "default" in param:
                    combo.setCurrentText(param["default"])
                self.dynamic_form.addWidget(combo)
                self.inputs[param["name"]] = combo

            elif ptype == "int":
                spinner = QSpinBox()
                spinner.setRange(param.get("min", 0), param.get("max", 9999))
                spinner.setValue(param.get("default", 0))
                self.dynamic_form.addWidget(spinner)
                self.inputs[param["name"]] = spinner

            elif ptype == "bool":
                chk = QCheckBox(param["label"])
                chk.setChecked(param.get("default", False))
                self.dynamic_form.addWidget(chk)
                self.inputs[param["name"]] = chk

            elif ptype == "file":
                row = QHBoxLayout()
                btn = QPushButton("Browse…")
                chosen = QLabel("No file selected")
                row.addWidget(btn)
                row.addWidget(chosen)
                self.dynamic_form.addLayout(row)

                def pick_file(p=param, lbl=chosen):
                    filt = p["dialog"].get("filter", "All Files (*)")
                    path, _ = QFileDialog.getOpenFileName(self, p["label"], "", filt)
                    if path:
                        lbl.setText(path)

                btn.clicked.connect(pick_file)
                self.inputs[param["name"]] = lambda lbl=chosen: lbl.text()

            else:
                # fallback text
                edit = QLineEdit()
                edit.setText(str(param.get("default", "")))
                self.dynamic_form.addWidget(edit)
                self.inputs[param["name"]] = edit

    def run_plugin(self):
        item = self.plugin_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Plugin Selected", "Select a plugin first.")
            return

        name = item.data(Qt.UserRole)
        kwargs: Dict[str, Any] = {}

        # Gather inputs
        for key, widget in self.inputs.items():
            if callable(widget):
                kwargs[key] = widget()
            elif isinstance(widget, QComboBox):
                kwargs[key] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                kwargs[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                kwargs[key] = widget.isChecked()
            elif hasattr(widget, "text"):
                kwargs[key] = widget.text()

        if self.json_checkbox.isChecked():
            kwargs["json_out"] = True

        self.output_console.clear()
        self.output_console.append(f">>> Running '{name}' …\n")

        worker = Worker(self.manager, name, kwargs)
        worker.finished.connect(self.on_worker_finished)
        worker.error.connect(self.on_worker_error)

        t = threading.Thread(target=worker.run, daemon=True)
        t.start()

    def on_worker_finished(self, text: str):
        self.output_console.append(text)

    def on_worker_error(self, err: str):
        self.output_console.append(f"ERROR: {err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AppGui()
    gui.show()
    sys.exit(app.exec())
