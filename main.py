main.py

from core.ingestor.collector import collect_sample 
from core.neural_engine.profiler import get_embedding
import tkinter as tk from tkinter
import filedialog, messagebox 
import json

class SentenialXGUI: def init(self, root): self.root = root self.root.title("SentenialX A.I. Threat Collector") self.root.geometry("600x400")

self.label = tk.Label(root, text="Choose a threat sample to analyze", font=("Helvetica", 14))
    self.label.pack(pady=20)

    self.select_button = tk.Button(root, text="Select File", command=self.load_sample)
    self.select_button.pack(pady=10)

    self.result_text = tk.Text(root, height=15, width=70)
    self.result_text.pack(pady=10)

def load_sample(self):
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    try:
        sample_data = collect_sample(file_path)
        with open(file_path, 'r', errors='ignore') as f:
            text = f.read(1000)
        embedding = get_embedding(text)

        result = {
            "sample_info": sample_data,
            "embedding_preview": embedding[:10].tolist()  # Truncate for display
        }
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, json.dumps(result, indent=2))
    except Exception as e:
        messagebox.showerror("Error", str(e))

if name == "main": root = tk.Tk() app = SentenialXGUI(root) root.mainloop()

