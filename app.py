from fastapi import FastAPI
import pickle
import gdown
import os

app = FastAPI()

MODEL_PATH = "models/misinformation_model"
GDRIVE_URL = "https://drive.google.com/drive/folders/1qfny70Of9yPJwE5JQuiKOXNE_VYrZV7o?usp=sharing"

# Load model from Google Drive (if not present)
if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    text = data["input"]
    output = model.predict([text])
    return {"prediction": output.tolist()}
