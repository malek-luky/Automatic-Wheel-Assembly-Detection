import numpy as np
import torch

from src.models.model import TireAssemblyLSTM

# HYPERPARAMETERS (must be same as in training)
SEQUENCE_LENGTH = 10
HIDDEN_LAYER_SIZE = 50
OUTPUT_SIZE = 1
NUM_FEATURES = 8
INPUT_SIZE = NUM_FEATURES
MODEL_NAME = 'model_20240113-213114.pth'

example_sequence = np.random.rand(SEQUENCE_LENGTH, NUM_FEATURES)
example_tensor = torch.tensor(example_sequence, dtype=torch.float).unsqueeze(0)


# To load the state dict, you need to re-create the model and load the state dict into it

loaded_model = TireAssemblyLSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)  # Re-create the model
loaded_model.load_state_dict(torch.load(f'models/{MODEL_NAME}'))

loaded_model.eval()

with torch.no_grad():
    logits = loaded_model(example_tensor)
    prediction_probability = torch.sigmoid(logits)  # Convert logits to probabilities

print("Prediction (probability):", prediction_probability.item())
