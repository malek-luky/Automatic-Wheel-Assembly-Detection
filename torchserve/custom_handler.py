import os
import torch
from src.models.model import TireAssemblyLSTM

# Create model object
INPUT_SIZE = 8
HIDDEN_LAYER_SIZE = 50
OUTPUT_SIZE = 1
model = TireAssemblyLSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

# Entry point function for the custom handler. This shiuld be enough if we only want to 
# get predictions by calling  'curl http://127.0.0.1:8080/predictions/TireAssemblyLSTM -T '
def handle(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # can be used with jit for optimization, but then the model must be saved from torch.jit.script(model)
        # model = torch.jit.load(model_pt_path) 
        model.load_state_dict(torch.load(model_pt_path))
        model.eval()

    #infer and return result
    else:
        # process input data
        data = convert_bytearray_to_tensor(data)
        
        # make predictions
        logits = model(data)
        
        # Convert logits to probabilities
        prediction_probability = torch.sigmoid(logits)  
        
        # return prediction as dictionary
        return [{"predictions": prediction_probability.item()}]

def convert_bytearray_to_tensor(bytearray_data):
    # Assuming the input is a list of dictionaries with 'body' key
    for item in bytearray_data:
        if 'body' in item:
            # Decode the bytearray to a string
            data_str = item['body'].decode('utf-8')

            # Split the string into lines
            lines = data_str.split('\n')

            # Convert lines to a list of lists of floats
            data = [[float(value) for value in line.split()] for line in lines if line]

            # Convert the list of lists to a NumPy array
            tensor = torch.tensor(data, dtype=torch.float)

            # Replace the 'body' value with the PyTorch tensor
            item['body'] = tensor
    data = bytearray_data[0]['body'].unsqueeze(0)
    return data

