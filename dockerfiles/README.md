In this modified Dockerfile, the conda env create command is used to create a Conda environment based on the environment.yml file. Then, the conda install command is used to install the dependencies specified in the requirements.txt file. The SHELL instruction is used to activate the Conda environment before running the subsequent commands. This approach allows you to reuse the cache from the last build, saving time and resources during the image building process. By following these steps, you can create a Dockerfile that leverages Conda for managing dependencies and efficiently reuses the cache during image builds.

We can start as many docker files as we want as long as they have original names docker run --name experiment1 trainer:latest

-The tad trainer:latest is not used in our case
-The file is called Dockerfile, which is the default name, therefore -f trainer.dockerfile is not needed to type
'docker build -f trainer.dockerfile . -t trainer:latest'

-To make sure only one docker file is running, tag --name {container_name} is not specified as well as the tag
'docker run --name {container_name} -v %cd%/models:/models/ trainer:latest'

Never mix pip and conda, if you want to use only pip (does not cross check with the packages of correct version) dont use conda, if you want to use conda, put all dependencies into environment.yml file and activate the codna environment in the last step.

Building docker using conda in repeat is 20000x faster (loading from cache does not work using only pip)

### Install the dependencies using pip
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

### Install the dependencies using conda
COPY environment.yml environment.yml #Copy Conda Environment File
RUN conda env create -f environment.yml #Create a Conda environment with dependencies
SHELL ["conda", "run", "-n", "DTU_ML_Ops", "/bin/bash", "-c"] #Last step!!! (activates the environment)
