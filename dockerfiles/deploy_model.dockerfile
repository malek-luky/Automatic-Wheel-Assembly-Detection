# Use a base image with Miniconda installed
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the container's working directory
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DTU_ML_Ops", "/bin/bash", "-c"]

# Copy the necessary project files into the container
COPY src ./src
COPY pyproject.toml .

# If you want to build the image locally, you need to download the service account key and set it as an environment variable 

#COPY vertex-sa.json ./vertex-sa.json
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/vertex-sa.json"

# Install the local package
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 80

# Set the command to run the FastAPI app using Uvicorn
CMD ["conda", "run", "-n", "DTU_ML_Ops", "uvicorn", "src.models.serve_model:app", "--host", "0.0.0.0", "--port", "80"]