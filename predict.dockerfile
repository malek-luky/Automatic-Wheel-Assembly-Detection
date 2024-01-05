# Base image
# Base image
FROM nvcr.io/nvidia/pytorch:23.07-py3 
# without CUDA use python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

# install dependencies
WORKDIR /
RUN pip install . --no-cache-dir #(1)

# training script
ENTRYPOINT ["python", "-u", "src/predict_model.py"]