import os
import shutil
import gdown
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
app = FastAPI()

# Define paths and Google Drive URL
MODEL_DIR = "models/misinformation_model"
FINAL_MODEL_DIR = os.path.join(MODEL_DIR, "models/misinformation_model")
GDRIVE_URL = "https://drive.google.com/drive/folders/1o5HZrgML4KSp7B-UODnfzMRS3xJUqjUw?usp=sharing"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive if not present
print("Checking model directory...")
if not os.listdir(MODEL_DIR):
    print("Downloading model files from Google Drive...")
    gdown.download_folder(GDRIVE_URL, output=MODEL_DIR, quiet=False, use_cookies=False)

# Check if the files are in the correct directory; if not, move them
if os.path.exists(FINAL_MODEL_DIR):
    print("Adjusting directory structure...")
    for file_name in os.listdir(FINAL_MODEL_DIR):
        src_path = os.path.join(FINAL_MODEL_DIR, file_name)
        dest_path = os.path.join(MODEL_DIR, file_name)

        # Only move files if they don't already exist
        if not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"File '{file_name}' already exists, skipping...")

    # Remove the empty nested directory after moving files
    shutil.rmtree(os.path.join(MODEL_DIR, "models"))

# Debugging: List directory structure
print("Directory structure after adjustment:")
for root, dirs, files in os.walk(MODEL_DIR):
    print(f"{root}/")
    for file in files:
        print(f"  {file}")

# Load tokenizer and model
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

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

@app.get("/")
def read_root():
    return {"message": "API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
