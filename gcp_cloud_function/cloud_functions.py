import wandb
from typing import List
import numpy as np
import os
import torch
from src.models.model import TireAssemblyLSTM
import functions_framework

def load_model():
    """
    Loads the model from the wandb artifact
    """
    print("Loading model from W&B")
    wandb.init(project="automatic-wheel-assembly-detection", entity="02476mlops")
    best_model = wandb.use_artifact("02476mlops/automatic-wheel-assembly-detection/mlops_model:latest")

    # Clean up the serve_model directory
    if os.path.exists("serve_model/"):
        shutil.rmtree("serve_model/")
    os.makedirs("serve_model/")

    best_model.download(root="serve_model/")

    # Find the model data in the directory
    for file in os.listdir("serve_model/"):
        if file.endswith(".pth"):
            model_name = file
            break

    state_dict = torch.load(f"serve_model/{model_name}")
    model = TireAssemblyLSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    model.load_state_dict(state_dict)
    return model

def predict(request):
    """
    Predicts the label for the given sequence via json in GCP Cloue Function
    WIP: Downloading torch as a requirement results in running out of memory
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        data = request_json['name']
    elif request_args and 'name' in request_args:
        data = request_args['name']
    else:
        return "No data found"


    model = load_model()


    # Convert input data to the appropriate format for your model
    example_sequence = np.array(data)

    # Check if input data is of correct shape
    if example_sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise HTTPException(status_code=400, detail=f"Input should be of shape ({SEQUENCE_LENGTH}, {INPUT_SIZE})")

    example_tensor = torch.tensor(example_sequence, dtype=torch.float).unsqueeze(0)

    # Use the loaded model for prediction
    model.eval()
    with torch.no_grad():
        logits = model(example_tensor)
        prediction_probability = torch.sigmoid(logits)  # Convert logits to probabilities
        label = 1 if prediction_probability.item() >= 0.5 else 0

    return {"prediction": label}