from fastapi import FastAPI
from pydantic import BaseModel
from .simulators.file_encryptor import simulate_encryption
from .simulators.ransom_note_generator import generate_note

app = FastAPI(title="Ransomware Simulator")

class SimInput(BaseModel):
    files: list[str]

@app.post("/simulate")
async def simulate_ransomware(input: SimInput):
    encrypted = simulate_encryption(input.files)
    note = generate_note()
    return {"encrypted_files": encrypted, "ransom_note": note}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
