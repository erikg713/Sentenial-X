import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit

class ThreatCollectorUI(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.output = QTextEdit()
        self.button = QPushButton("Run Threat Collection")
        self.button.clicked.connect(self.run_collector)
        layout.addWidget(self.button)
        layout.addWidget(self.output)
        self.setLayout(layout)

    def run_collector(self):
        self.output.append("Collecting threats...")
        # run your actual code here (e.g., self.output.append(str(run_threat_collection()))
# Dummy implementations for demo; replace with your actual imports
def collect_sample(file_path, suspicious_strings=None):
    # Simulate metadata extraction & suspicious strings detection
    return {
        "file": file_path,
        "matched_suspicious_strings": ["powershell.exe -EncodedCommand"] if suspicious_strings else [],
        "cve_ids": ["CVE-2023-12345"]
    }

def get_embedding(text):
    # Simulate embedding generation: fixed-length vector
    np.random.seed(len(text))
    return np.random.rand(64).tolist()

class ThreatCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX Threat Collector")
        self.root.geometry("900x700")

        self.current_embedding = None
        self.last_result = None
        self.dark_mode = True

        self.setup_ui()
        self.set_theme()

    def setup_ui(self):
        # Menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        opt_menu = tk.Menu(menu, tearoff=0)
        opt_menu.add_command(label="Toggle Dark/Light Mode", command=self.toggle_theme)
        opt_menu.add_command(label="Clear Logs", command=self.clear_logs)
        opt_menu.add_separator()
        opt_menu.add_command(label="Exit", command=self.root.quit)
        menu.add_cascade(label="Options", menu=opt_menu)

        # Buttons
        tk.Button(self.root, text="Select Threat Sample", command=self.load_sample_dialog).pack(pady=5)
        tk.Button(self.root, text="Visualize Embedding", command=self.visualize_embedding).pack(pady=5)
        tk.Button(self.root, text="Save Last Result", command=self.save_result).pack(pady=5)

        # Result output
        tk.Label(self.root, text="Analysis Result:").pack(anchor="w")
        self.result_text = scrolledtext.ScrolledText(self.root, height=12)
        self.result_text.pack(fill="both", expand=False, padx=5, pady=5)

        # Log output
        tk.Label(self.root, text="Logs:").pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(self.root, height=8)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.set_theme()

    def set_theme(self):
        bg = "#1e1e1e" if self.dark_mode else "#f0f0f0"
        fg = "#ffffff" if self.dark_mode else "#000000"
        self.root.config(bg=bg)
        self.result_text.config(bg=bg, fg=fg, insertbackground=fg)
        self.log_text.config(bg=bg, fg=fg, insertbackground=fg)

    def load_sample_dialog(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_sample(file_path)

    def load_sample(self, file_path):
        try:
            self.log(f"Loading sample: {file_path}")
            sample_data = collect_sample(file_path, suspicious_strings=True)
            with open(file_path, "r", errors="ignore") as f:
                text = f.read(1000)
            embedding = get_embedding(text)
            self.current_embedding = embedding
            self.last_result = {
                "sample_info": sample_data,
                "embedding_preview": embedding[:10],
            }
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(self.last_result, indent=2))
            self.log("Sample loaded and analyzed successfully.")
        except Exception as e:
            self.log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

    def visualize_embedding(self):
        if not self.current_embedding:
            messagebox.showinfo("Info", "Load a sample first to visualize embedding.")
            return
        try:
            vec = np.array(self.current_embedding).reshape(1, -1)
            if vec.shape[1] < 2:
                messagebox.showwarning("Visualization", "Embedding too small to visualize.")
                return
            pca = PCA(n_components=2)
            coords = pca.fit_transform(vec)
            plt.figure(figsize=(5,4))
            plt.scatter(coords[:,0], coords[:,1], c='purple')
            plt.title("PCA of Threat Embedding")
            plt.show()
        except Exception as e:
            self.log(f"[ERROR] Visualization failed: {e}")
            messagebox.showerror("Error", str(e))

    def save_result(self):
        if not self.last_result:
            messagebox.showinfo("Info", "No result to save yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.last_result, f, indent=2)
                self.log(f"Result saved to {path}")
            except Exception as e:
                self.log(f"[ERROR] Saving failed: {e}")
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ThreatCollectorGUI(root)
    root.mainloop()

