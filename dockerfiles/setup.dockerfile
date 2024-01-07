# Base image
FROM python:3.11-slim
# with CUDA use nvcr.io/nvidia/pytorch:23.07-py3 

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

# training script - we will use script to run it instead
# ENTRYPOINT ["python", "-u", "src/train_model.py"]