from fastapi import FastAPI
from inference import detect_prompt
from schema import PromptRequest

app = FastAPI(title="Jailbreak Detector")

@app.post("/detect")
def detect(request: PromptRequest):
    return detect_prompt(request.prompt)

