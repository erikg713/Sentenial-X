import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import json
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding
from utils.logger import logger
from gui.visualizer.Realtimethreats import RealTimeThreatsPanel

self.threat_panel = RealTimeThreatsPanel(self.content_frame)
self.threat_panel.pack(fill="both", expand=True)

class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("950x800")  # Increased size
        self.dark_mode = True
        self.cve_db = self.load_cve_database("cve_db.json")
        self.suspicious_strings = [
            "powershell.exe -EncodedCommand",
            "http://malicious.com",
            "registry key HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\Run",
            "function CreateRemoteThread",
            "api_peering",
            ".exe", ".dll", ".vbs", ".js"
        ]
        self.current_embedding = None

        self.setup_ui()
        self.bind_drag_and_drop()
        self.last_result = None

    def load_cve_database(self, db_path):
        try:
            with open(db_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.display_log(f"Warning: CVE database not found at {db_path}")
            return {}
        except json.JSONDecodeError:
            self.display_log(f"Error: Could not decode JSON from {db_path}")
            return {}
        return {}

    def setup_ui(self):
        self.set_theme()

        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        options_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Toggle Dark/Light Mode", command=self.toggle_theme)
        options_menu.add_command(label="Clear Logs", command=self.clear_log_output)
        options_menu.add_separator()
        options_menu.add_command(label="Exit", command=self.root.quit)

        self.label = tk.Label(self.root, text="Drop or Select a Threat Sample", font=("Helvetica", 14), bg=self.bg_color, fg=self.fg_color)
        self.label.pack(pady=10)

        self.select_button = tk.Button(self.root, text="Select File", command=self.load_sample_dialog, bg=self.button_bg, fg=self.fg_color)
        self.select_button.pack(pady=5)

        self.visualize_button = tk.Button(self.root, text="Visualize Embedding", command=self.handle_visualization, bg=self.button_bg, fg=self.fg_color)
        self.visualize_button.pack(pady=5)

        self.save_button = tk.Button(self.root, text="Save Result", command=self.save_result, bg=self.button_bg, fg=self.fg_color)
        self.save_button.pack(pady=5)

        # Result Output
        result_label = tk.Label(self.root, text="Analysis Result:", font=("Helvetica", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        result_label.pack(pady=5, anchor="w")
        self.result_text = self._add_scrollable_textbox(self.root, height=15)

        # Suspicious Strings Output
        suspicious_label = tk.Label(self.root, text="Suspicious Strings Found:", font=("Helvetica", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        suspicious_label.pack(pady=5, anchor="w")
        self.suspicious_text = self._add_scrollable_textbox(self.root, height=5, bg="#f0f8ff" if not self.dark_mode else "#303030", fg="#00008b" if not self.dark_mode else "#add8e6")

        # CVE Display Panel
        cve_label = tk.Label(self.root, text="Matched CVE Information:", font=("Helvetica", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        cve_label.pack(pady=5, anchor="w")
        self.cve_panel = self._add_scrollable_textbox(self.root, height=8, bg="#f5f5f5" if not self.dark_mode else "#333333", fg="#000000" if not self.dark_mode else "#dcdcdc")

        # Log Output
        self.log_frame = tk.Frame(self.root, bg=self.bg_color)
        self.log_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(self.log_frame, text="Logs:", bg=self.bg_color, fg=self.fg_color, anchor="w").pack()
        self.log_text = self._add_scrollable_textbox(self.log_frame, height=6)

    def _add_scrollable_textbox(self, parent, height=10, width=80, bg=None, fg=None):
        text_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=height, width=width, bg=bg or self.entry_bg, fg=fg or self.fg_color, insertbackground=fg or self.fg_color)
        text_area.pack(fill=tk.BOTH, expand=True)
        return text_area

    def bind_drag_and_drop(self):
        self.root.bind("<Drag>", lambda e: "break")
        self.root.bind("<DragEnter>", lambda e: "break")
        self.root.bind("<Drop>", self.handle_drop)

    def handle_drop(self, event):
        try:
            file_path = event.data.strip('{}')
            self.load_sample(file_path)
        except Exception as e:
            self.display_error(e)

    def load_sample_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.load_sample(file_path)

    def load_sample(self, file_path):
        if not os.path.exists(file_path):
            self.display_log(f"Invalid file path: {file_path}")
            return

        if os.path.getsize(file_path) > 5 * 1024 * 1024:
            messagebox.showwarning("File too large", "Please select a smaller file (under 5MB).")
            return

        try:
            self.display_log(f"Collecting sample: {file_path}")
            sample_data = collect_sample(file_path, suspicious_strings=self.suspicious_strings)

            with open(file_path, 'r', errors='ignore') as f:
                text = f.read(1000)

            embedding = get_embedding(text)
            self.current_embedding = embedding
            embedding_np = np.array(embedding)

            result = {
                "sample_info": sample_data,
                "embedding_preview": embedding[:10].tolist(),
                "embedding_info": {
                    "length": len(embedding),
                    "norm": np.linalg.norm(embedding_np).item()
                }
            }

            self.last_result = result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(result, indent=2))

            # Display suspicious strings
            self.display_suspicious_strings(sample_data.get("matched_suspicious_strings", []))

            # Match and display CVEs
            cve_ids = sample_data.get("cve_ids", [])
            matched_cves = self.match_cves(cve_ids)
            self.display_cve_info(matched_cves)

            self.display_log("Sample analysis complete.")
        except Exception as e:
            self.display_error(e)

    def display_suspicious_strings(self, strings):
        self.suspicious_text.delete(1.0, tk.END)
        if strings:
            self.suspicious_text.insert(tk.END, "\n".join(strings))
        else:
            self.suspicious_text.insert(tk.END, "No suspicious strings found.")

    def match_cves(self, cve_ids):
        matched = {}
        for cve_id in cve_ids:
            if cve_id in self.cve_db:
                matched[cve_id] = self.cve_db[cve_id]
        return matched

    def display_cve_info(self, matched_cves):
        self.cve_panel.delete(1.0, tk.END)
        if matched_cves:
            for cve_id, data in matched_cves.items():
                self.cve_panel.insert(tk.END, f"{cve_id} - Severity: {data.get('severity', 'N/A')}\n")
                self.cve_panel.insert(tk.END, f"Description: {data.get('description', 'No description available')}\n")
                self.cve_panel.insert(tk.END, f"Published: {data.get('published', 'N/A')}\n\n")
        else:
            self.cve_panel.insert(tk.END, "No matching CVEs found in the database.")

    def handle_visualization(self):
        if self.current_embedding is not None:
            self.visualize_embedding(self.current_embedding)
        else:
            messagebox.showwarning("No Embedding", "Load a sample first to generate an embedding.")

    def visualize_embedding(self, embedding):
        try:
            X = np.array(embedding).reshape(1, -1)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            plt.figure(figsize=(6, 5))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], color='#a020f0')  # Vibrant purple
            plt.title("PCA Visualization of Embedding", color=self.fg_color)
            plt.xlabel("Principal Component 1", color=self.fg_color)
            plt.ylabel("Principal Component 2", color=self.fg_color)
            plt.grid(True, color='#555555')
            plt.gca().set_facecolor(self.bg_color)
            plt.tick_params(colors=self.fg_color)
            plt.show()
        except Exception as e:
            messagebox.showerror("Visualization Error", str(e))

    def save_result(self):
        if not self.last_result:
            messagebox.showinfo("Nothing to save", "No analysis result available.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(self.last_result, f, indent=2)
                self.display_log(f"Result saved to: {save_path}")
            except Exception as e:
                self.display_error(e)

    def display_log(self, msg):
        logger.info(msg)
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)

    def display_error(self, e):
        logger.error(str(e))
        self.display_log(f"[ERROR] {str(e)}")
        messagebox.showerror("Error", str(e))

    def clear_log_output(self):
        self.log_text.delete(1.0, tk.END)
        self.display_log("Log cleared.")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.set_theme()
        self.root.configure(bg=self.bg_color)
        self.label.config(bg=self.bg_color, fg=self.fg_color)
        self.select_button.config(bg=self.button_bg, fg=self.fg_color)
        self.visualize_button.config(bg=self.button_bg, fg=self.fg_color)
        self.save_button.config(bg=self.button_bg, fg=self.fg_color)
        self.result_text.config(bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color)
        self.suspicious_text.config(bg="#f0f8ff" if not self.dark_mode else "#303030", fg="#00008b" if not self.dark_mode else "#add8e6", insertbackground=self.fg_color)
        self.cve_panel.config(bg="#f5f5f5" if not self.dark_mode else "#333333", fg="#000000" if not self.dark_mode else "#dcdcdc", insertbackground=self.fg_color)
        self.log_frame.config(bg=self.bg_color)
        for widget in self.log_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(bg=self.bg_color, fg=self.fg_color)
        self.log_text.config(bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color)

    def set_theme(self):
        if self.dark_mode:
            self.bg_color = "#1e1e1e"
            self.fg_color = "#ffffff"
            self.entry_bg = "#2e2e2e"
            self.button_bg = "#444444"
        else:
            self.bg_color = "#f0f0f0"
            self.fg_color = "#000000"
            self.entry_bg = "#ffffff"
            self.button_bg = "#dddddd"

import tkinter as tk
from tkinter import filedialog, messagebox
import json
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding


class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("800x600")
        self.root.configure(bg="#1c1c1c")

        self.last_embedding = None

        self.init_dashboard()

    def init_dashboard(self):
        # Instruction label
        self.label = tk.Label(self.root, text="Choose a threat sample to analyze", font=("Helvetica", 14), bg="#1c1c1c", fg="#ffffff")
        self.label.pack(pady=15)

        # Select file button
        self.select_button = tk.Button(self.root, text="Select File", command=self.load_sample, bg="#0d6efd", fg="white", font=("Helvetica", 11))
        self.select_button.pack(pady=10)

        # Result JSON text output
        self.result_text = tk.Text(self.root, height=12, width=90, bg="#2b2b2b", fg="#00ff00", insertbackground="#00ff00")
        self.result_text.pack(pady=10)

        # Similarity Label
        self.similarity_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#1c1c1c", fg="cyan")
        self.similarity_label.pack(pady=5)

        # Embedding chart frame
        self.chart_frame = tk.Frame(self.root, bg="#1c1c1c")
        self.chart_frame.pack(pady=10)

    def load_sample(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        try:
            # Collect structured sample metadata
            sample_data = collect_sample(file_path)

            # Extract text from the file
            with open(file_path, 'r', errors='ignore') as f:
                text = f.read(1000)

            # Generate embedding
            embedding = get_embedding(text)

            # Cosine similarity comparison
            similarity = None
            if self.last_embedding is not None:
                try:
                    similarity = self.cosine_similarity(self.last_embedding, embedding)
                    self.similarity_label.config(text=f"Similarity with last sample: {similarity:.4f}")
                except Exception as e:
                    self.similarity_label.config(text=f"Similarity error: {e}")
            else:
                self.similarity_label.config(text="No previous sample to compare.")

            self.last_embedding = embedding  # Save for next sample

            # Prepare result preview
            result = {
                "sample_info": sample_data,
                "embedding_preview": embedding[:10].tolist()  # Truncated preview
            }

            # Show result JSON
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(result, indent=2))

            # Show embedding as chart
            self.show_embedding_chart(embedding)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_embedding_chart(self, vector):
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Display first 32 values
        vector_slice = vector[:32]

        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ax.bar(range(len(vector_slice)), vector_slice, color='skyblue')
        ax.set_title("Embedding Preview (First 32 Values)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        fig.tight_layout()

        chart = FigureCanvasTkAgg(fig, master=self.chart_frame)
        chart.draw()
        chart.get_tk_widget().pack()

    def cosine_similarity(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel

# Import your widgets here (assumed to be saved in gui/dashboard/widgets/)
from gui.dashboard.widgets.threat_table import ThreatTable
from gui.dashboard.widgets.telemetry_viewer import TelemetryViewer
from gui.dashboard.widgets.simulation_controls import SimulationControls
from gui.dashboard.widgets.report_card import ReportCard
from gui.dashboard.widgets.attack_graph import AttackGraph

# Dummy sample threat data for demonstration
SAMPLE_THREATS = [
    {"id": "T-1001", "severity": "High", "name": "Ransomware", "description": "Encrypts files", "tags": ["ransomware", "encryption"]},
    {"id": "T-1002", "severity": "Critical", "name": "Zero-Day", "description": "Unknown exploit", "tags": ["zero-day", "exploit"]},
    {"id": "T-1003", "severity": "Medium", "name": "Phishing", "description": "Credential theft", "tags": ["phishing", "social engineering"]}
]

class SentenialXDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Threats Tab
        threat_tab = QWidget()
        threat_layout = QVBoxLayout()
        self.threat_table = ThreatTable()
        self.threat_table.load_data(SAMPLE_THREATS)
        threat_layout.addWidget(self.threat_table)
        threat_tab.setLayout(threat_layout)
        self.tabs.addTab(threat_tab, "Threats")

        # Telemetry Tab
        telemetry_tab = QWidget()
        telemetry_layout = QVBoxLayout()
        self.telemetry_viewer = TelemetryViewer()
        telemetry_layout.addWidget(self.telemetry_viewer)
        telemetry_tab.setLayout(telemetry_layout)
        self.tabs.addTab(telemetry_tab, "Telemetry")

        # Simulation Controls Tab
        sim_tab = QWidget()
        sim_layout = QVBoxLayout()
        self.sim_controls = SimulationControls()
        sim_layout.addWidget(self.sim_controls)
        sim_tab.setLayout(sim_layout)
        self.tabs.addTab(sim_tab, "Simulation Controls")

        # Reports Tab
        report_tab = QWidget()
        report_layout = QVBoxLayout()
        self.report_card = ReportCard(title="Daily Security Report", summary="No critical threats detected.")
        report_layout.addWidget(self.report_card)
        report_tab.setLayout(report_layout)
        self.tabs.addTab(report_tab, "Reports")

        # Attack Graph Tab
        graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        self.attack_graph = AttackGraph()
        graph_layout.addWidget(self.attack_graph)
        graph_tab.setLayout(graph_layout)
        self.tabs.addTab(graph_tab, "Attack Graph")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentenialXDashboard()
    window.show()
    sys.exit(app.exec())

