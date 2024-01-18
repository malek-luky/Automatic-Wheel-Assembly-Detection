# Local deployment using Torchserve

## Installation
Follow [these instructions](https://github.com/pytorch/serve) to download torchserve based on your OS. 


Install `torch-model-archiver` as follows:

```bash
pip install torch-model-archiver
```

Install `curl` for making requests
```bash
pip install pycurl
```

## Torch Model Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `model-archiver`.

First make a new folder for storing the arhive
```bash
mkdir model_store
```
Then call this to create a model (with .mar extension) inside the folder model_store.
```bash

torch-model-archiver     --model-name TireAssemblyLSTM --version 1.0  --serialized-file < path-to-model.pth > --export-path torchserve/model_store --handler /torchserve/custom_handler:handle 

```

Default hadnlers (image_classifier, object_detector, text_classifier, image_segmenter) were not suitable for our model, so a custom one was implemeted. The entry point function os called `handle` in `custom_handler.py`.


## Run the model server

To run the server listening on port 8080 call this in one terminal

```bash
torchserve --start --ncs --model-store model_store --models model_store/TireAssemblyLSTM.mar
```

## Make predictions

To make predictions use the following curl request with data, as for example it is provided in example_tensor.txt 
```bash 
curl http://127.0.0.1:8080/predictions/TireAssemblyLSTM -T example_tensor.txt 
```
The server is going to reply with the predictions from the model:
```bash 
{
  "predictions": 0.0381014384329319
}
```

## Stop serving
When you are done, terminate the server with
```bash
torchserve --stop
```