import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import json
import numpy as np

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding
from utils.logger import logger  # Make sure this is set up

class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("900x700")  # Slightly larger
        self.dark_mode = True
        self.cve_db = self.load_cve_database("cve_db.json")

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

        self.save_button = tk.Button(self.root, text="Save Result", command=self.save_result, bg=self.button_bg, fg=self.fg_color)
        self.save_button.pack(pady=5)

        # Result Output
        result_label = tk.Label(self.root, text="Analysis Result:", font=("Helvetica", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        result_label.pack(pady=5, anchor="w")
        self.result_text = self._add_scrollable_textbox(self.root, height=12)

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

            # Match and display CVEs
            cve_ids = sample_data.get("cve_ids", [])
            matched_cves = self.match_cves(cve_ids)
            self.display_cve_info(matched_cves)

            self.display_log("Sample analysis complete.")
        except Exception as e:
            self.display_error(e)

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
        self.save_button.config(bg=self.button_bg, fg=self.fg_color)
        self.result_text.config(bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = SentenialXGUI(root)
    root.mainloop()
