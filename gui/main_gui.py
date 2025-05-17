import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import numpy as np

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding
from utils.logger import logger

class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("800x600")
        self.dark_mode = True

        self.setup_ui()
        self.bind_drag_and_drop()
        self.last_result = None

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

        self.save_button = tk.Button(self.root, text="Save Result", command=self.save_result, bg=self.button_bg, fg=self.fg_color)
        self.save_button.pack(pady=5)

        self.result_frame = tk.Frame(self.root, bg=self.bg_color)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.result_text = self._add_scrollable_textbox(self.result_frame)

        self.log_frame = tk.Frame(self.root, bg=self.bg_color)
        self.log_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(self.log_frame, text="Logs:", bg=self.bg_color, fg=self.fg_color).pack(anchor="w")

        self.log_text = self._add_scrollable_textbox(self.log_frame)
        self.log_text.configure(height=6)
    
    def _add_scrollable_textbox(self, parent):
        scrollbar = tk.Scrollbar(parent)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget = tk.Text(parent, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                              bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        return text_widget

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
            sample_data = collect_sample(file_path)

            with open(file_path, 'r', errors='ignore') as f:
                text = f.read(1000)

            embedding = get_embedding(text)
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
            self.display_log("Sample analysis complete.")
        except Exception as e:
            self.display_error(e)

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
from tkinter import filedialog, messagebox, scrolledtext
import json
import os

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding

class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Choose a threat sample to analyze", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select File", command=self.load_sample)
        self.select_button.pack(pady=5)

        # Result JSON Output
        self.result_text = scrolledtext.ScrolledText(root, height=15, width=90)
        self.result_text.pack(pady=10)

        # CVE Display Panel
        self.cve_label = tk.Label(root, text="Matched CVE Information", font=("Helvetica", 12, "bold"))
        self.cve_label.pack()
        self.cve_panel = scrolledtext.ScrolledText(root, height=10, width=90, bg="#f5f5f5")
        self.cve_panel.pack(pady=5)

    def load_sample(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        try:
            sample_data = collect_sample(file_path)
            with open(file_path, 'r', errors='ignore') as f:
                text = f.read(1000)
            embedding = get_embedding(text)

            # Load CVE DB and match
            matched_cves = self.match_cves(sample_data.get("cve_ids", []))

            # Display results
            result = {
                "sample_info": sample_data,
                "embedding_preview": embedding[:10].tolist()
            }

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(result, indent=2))

            self.cve_panel.delete(1.0, tk.END)
            if matched_cves:
                for cve_id, data in matched_cves.items():
                    self.cve_panel.insert(tk.END, f"{cve_id} - {data['severity']}\n")
                    self.cve_panel.insert(tk.END, f"{data['description']}\nPublished: {data['published']}\n\n")
            else:
                self.cve_panel.insert(tk.END, "No matching CVEs found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def match_cves(self, cve_ids):
        try:
            with open("cve_db.json", "r") as f:
                cve_db = json.load(f)
        except FileNotFoundError:
            return {}

        matched = {}
        for cve_id in cve_ids:
            if cve_id in cve_db:
                matched[cve_id] = cve_db[cve_id]
        return matched

if __name__ == "__main__":
    root = tk.Tk()
    app = SentenialXGUI(root)
    root.mainloop()
