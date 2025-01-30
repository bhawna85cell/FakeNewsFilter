from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown
import os

app = FastAPI()

# Define paths and Google Drive URL
MODEL_DIR = "./models/misinformation_model"
MODEL_FILE = "model.safetensors"
GDRIVE_URL = "https://drive.google.com/drive/folders/1o5HZrgML4KSp7B-UODnfzMRS3xJUqjUw?usp=sharing"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


print("Downloading model from Google Drive...")
gdown.download_folder(GDRIVE_URL, quiet=False, output=MODEL_DIR)


# Debug: Print the directory structure
print("Directory structure after download:")
for root, dirs, files in os.walk(MODEL_DIR):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")

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
