from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch
import wandb
from src.models.model import TireAssemblyLSTM
from src.helper.gcp_utils import get_secret
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Function to load the model


def load_model():
    WANDB_API_KEY = get_secret('wheel-assembly-detection', 'WANDB_API_KEY')
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    wandb.init(project="automatic-wheel-assembly-detection", entity="02476mlops")
    best_model = wandb.use_artifact("02476mlops/automatic-wheel-assembly-detection/mlops_model:latest")
    best_model.download(root="serve_model/")

    # Find the model name in the directory
    for file in os.listdir("serve_model/"):
        if file.endswith(".pth"):  # Assuming your model file has a .pt extension
            model_name = file
            break

    state_dict = torch.load(f"serve_model/{model_name}")
    model = TireAssemblyLSTM(8, 50, 1)
    model.load_state_dict(state_dict)
    return model


# Global dictionary to store the model
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and store the model
    models["tire_assembly_lstm"] = load_model()

    # Yield control back to FastAPI
    yield

    # Cleanup, if necessary
    models.clear()

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Background task to refresh the model


async def refresh_model():
    while True:
        await asyncio.sleep(3600)  # Check for a new model every hour
        models["tire_assembly_lstm"] = load_model()

# Prediction endpoint

# Assuming you've defined these somewhere or import them
SEQUENCE_LENGTH = 10  # Define or import your sequence length
NUM_FEATURES = 8     # Define or import your number of features
INPUT_SIZE = NUM_FEATURES       # Define or import your input size
HIDDEN_LAYER_SIZE = 50  # Define or import your hidden layer size
OUTPUT_SIZE = 1      # Define or import your output size

# Define a request model for your API


class PredictionRequest(BaseModel):
    sequence: list[list[float]]  # Nested list to represent the sequence


@app.post("/predict")
async def predict(request: PredictionRequest):
    # Convert input data to the appropriate format for your model
    example_sequence = np.array(request.sequence)

    # Check if input data is of correct shape
    if example_sequence.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
        raise HTTPException(status_code=400, detail=f"Input should be of shape ({SEQUENCE_LENGTH}, {NUM_FEATURES})")

    example_tensor = torch.tensor(example_sequence, dtype=torch.float).unsqueeze(0)

    # Use the loaded model for prediction
    models["tire_assembly_lstm"].eval()
    with torch.no_grad():
        logits = models["tire_assembly_lstm"](example_tensor)
        prediction_probability = torch.sigmoid(logits)  # Convert logits to probabilities

    return {"prediction_probability": prediction_probability.item()}
