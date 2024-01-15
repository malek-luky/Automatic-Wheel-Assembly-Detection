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
# Copy the Git repository files
COPY .git ./.git

# Install the local package
RUN pip install -e .

# Copy the DVC files
COPY data.dvc .
COPY .dvc ./.dvc

# Run DVC pull to fetch the data
RUN dvc pull

# Add environment activation command to .bashrc (so when container starts, the environment is activated)
RUN echo "source activate DTU_ML_Ops" >> ~/.bashrc

# The command to run when the container starts
# Keeps the container running, you can then run docker exec -it <container_id> /bin/bash to get a bash shell inside the container
CMD tail -f /dev/null 

