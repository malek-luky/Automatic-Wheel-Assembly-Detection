# Base image
FROM python:3.10.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files
WORKDIR /home/mlops
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY .git/ .git/

# install dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-cache-dir
RUN dvc pull

# run
# ENTRYPOINT ["python", "-u", "src/data/make_dataset.py"]