from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown
import os

app = FastAPI()

# Define paths and Google Drive URL
MODEL_DIR = "./models/misinformation_model"
MODEL_FILE = "model.safetensors"
GDRIVE_FILE_ID = "1qfny70Of9yPJwE5JQuiKOXNE_VYrZV7o"  # Extracted from the folder link
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive if not present
if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, os.path.join(MODEL_DIR, MODEL_FILE), quiet=False)
else:
    print("Model already exists locally.")

# Load tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

@app.post("/predict")
def predict(data: dict):
    """
    Endpoint to predict the class of the input text.

    Input:
    {
        "input": "Text to classify"
    }

    Output:
    {
        "prediction": class_index
    }
    """
    text = data["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return {"prediction": prediction}
