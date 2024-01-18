import os
import shutil
from contextlib import asynccontextmanager

import numpy as np
import torch
import wandb
from fastapi import FastAPI, HTTPException
from fastapi_restful.tasks import repeat_every
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.helper.gcp_utils import get_secret
from src.models.model import TireAssemblyLSTM

# Global variables

config = OmegaConf.load("src/models/config/default_config.yaml")
hparams = config

SEQUENCE_LENGTH = hparams.sequence_length
INPUT_SIZE = hparams.input_size
HIDDEN_LAYER_SIZE = hparams.hidden_layer_size
OUTPUT_SIZE = hparams.output_size

WANDB_API_KEY = get_secret("wheel-assembly-detection", "WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Global dictionary to store the model
models = {}


def load_model():
    print("Loading model from W&B")
    wandb.init(project="automatic-wheel-assembly-detection", entity="02476mlops")
    best_model = wandb.use_artifact("02476mlops/automatic-wheel-assembly-detection/mlops_model:latest")

    # Clean up the serve_model directory
    if os.path.exists("serve_model/"):
        shutil.rmtree("serve_model/")
    os.makedirs("serve_model/")

    best_model.download(root="serve_model/")

    # Find the model name in the directory
    for file in os.listdir("serve_model/"):
        if file.endswith(".pth"):
            model_name = file
            break

    state_dict = torch.load(f"serve_model/{model_name}")
    model = TireAssemblyLSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    model.load_state_dict(state_dict)
    return model


@repeat_every(seconds=60 * 60 * 6)  # repeat every 6 hours
async def update_model_periodically():
    print("Checking for a new model")
    models["tire_assembly_lstm"] = load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and store the model
    await update_model_periodically()

    # Yield control back to FastAPI
    yield

    # Cleanup, if necessary
    models.clear()


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    sequence: list[list[float]]  # Nested list to represent the sequence


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make a prediction for a given sequence.

    The sequence should be a nested list of floats with shape (SEQUENCE_LENGTH = 10, INPUT_SIZE = 8).

    Example:

    ```
    {
        "sequence": [
            [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2],
            ...
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2]
        ]
    }
    ```

    The response will be a JSON object with the following format:

        ```
        {
            "prediction": 0
        }
        ```

    Where the value of `prediction` is either 0 or 1.
    """

    # Convert input data to the appropriate format for your model
    example_sequence = np.array(request.sequence)

    # Check if input data is of correct shape
    if example_sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise HTTPException(status_code=400, detail=f"Input should be of shape ({SEQUENCE_LENGTH}, {INPUT_SIZE})")

    example_tensor = torch.tensor(example_sequence, dtype=torch.float).unsqueeze(0)

    # Use the loaded model for prediction
    models["tire_assembly_lstm"].eval()
    with torch.no_grad():
        logits = models["tire_assembly_lstm"](example_tensor)
        prediction_probability = torch.sigmoid(logits)  # Convert logits to probabilities
        label = 1 if prediction_probability.item() >= 0.5 else 0

    return {"prediction": label}


# healthcheck


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


# root


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Automatic Wheel Assembly Detection Model API go to /docs or call the /predict endpoint!"
    }
