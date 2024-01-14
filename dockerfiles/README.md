In this modified Dockerfile, the conda env create command is used to create a Conda environment based on the environment.yml file. Then, the conda install command is used to install the dependencies specified in the requirements.txt file. The SHELL instruction is used to activate the Conda environment before running the subsequent commands. This approach allows you to reuse the cache from the last build, saving time and resources during the image building process. By following these steps, you can create a Dockerfile that leverages Conda for managing dependencies and efficiently reuses the cache during image builds.

## NOTES
- The file name convention is name:tag  (e.g. trainer:latest)
- We can start as many docker files as we want as long as they have original names docker run --name experiment1 trainer:latest
- If the file is called Dockerfile, which is the default name, then -f trainer.dockerfile is not needed to type
- If dvc pull is not working, try to download dvc as a conda package, not using pip
- Building docker using conda in repeat is 20000x faster (loading from cache does not work using only pip on Windows)
- Never mix pip and conda, if you want to use only pip (does not cross check with the packages of correct version) dont use conda, if you want to use conda, put all dependencies into environment.yml file and activate the codna environment in the last step.
- Conda can be very tricky while using in docker, after 7 hours I conclude that you have to be a masochist to use conda in docker. But if it works, great

## BUILDING PROCESS
- Normally we would build and store the docker image on GCP
- However there were issues during the building process
- Therefore we build it using GitHub workflow and then send it to GCP
- Now the stores image on GCP can be immediately deployed (both online and locally)

## COMMANDS
- `docker build -f train_wheel_assembly_detection.dockerfile . -t trainer:latest` (builds trainer image)
- `docker build -f conda_wheel_assembly_detection.dockerfile . -t conda:latest` (builds conda image)
- `docker run --name instance1 -it --entrypoint /bin/bash train:latest` (interactive mode)
- `docker run --name instance1 trainer:latest` (just run the docker)
- `docker run --name instance1 -v %cd%/models:/models/ trainer:latest` (automatically copies the created files to local machine, otherwise they will be only inside the docker)
- `docker rm <container_id>` (removes the specified container)





