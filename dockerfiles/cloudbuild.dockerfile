# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the necessary project files into the container
COPY ../src ./src
COPY ../pyproject.toml .
# Copy the Git repository files
COPY ../.git ./.git

# Install the local package
RUN pip install dvc
RUN pip install dvc-gs

# Copy the DVC files
COPY ../data.dvc .
COPY ../.dvc ./.dvc

# Run DVC pull to fetch the data
RUN dvc pull --verbose

# The command to run when the container starts
#CMD ["/bin/bash"]
