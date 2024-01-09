# Base image
FROM continuumio/miniconda3
WORKDIR /app

# Create a Conda environment with dependencies
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DTU_ML_Ops", "/bin/bash", "-c"]

# Copy other files
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data.dvc data.dvc
COPY /.dvc .
COPY /.git .

# Demonstrate the environment is activated:
RUN echo "Make sure torch is installed:"
RUN python -c "import torch"
SHELL ["conda", "dvc pull", "-n", "DTU_ML_Ops", "/bin/bash", "-c"]

# Run the application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "DTU_ML_Ops", "python", "src/data/make_dataset.py"]