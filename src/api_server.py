from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("models/optimized")
model = AutoModelForCausalLM.from_pretrained("models/optimized", device_map="auto")

class Query(BaseModel):
    prompt: str
    http_session: str = None  # optional serialized HTTP

@app.post("/infer/")
def infer(q: Query):
    inputs = tokenizer(q.prompt, return_tensors="pt").to(model.device)
    # If HTTP fusion, preprocess and inject
    if q.http_session:
        # serialize and get embedding
        pass
    output = model.generate(**inputs, max_new_tokens=128)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

@app.get("/healthz")
 def health():
-    return {"status": "ok"}
+    return {
+        "status": "ok",
+        "name": "Sentenial-X A.I.",
+        "tagline": "Crafted for resilience. Engineered for vengeance."
+    }
