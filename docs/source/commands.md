## Commands

### Docker
- `docker build -f dockerfiles/train_model_vm.dockerfile . -t trainer_vm:latest`
- `docker run --name instance1 -it --entrypoint /bin/bash trainer_vm:latest` (interactive mode)
- `docker run --name instance1 trainer_vm:latest` (just run the docker)
- `docker run --name instance1 -v %cd%/models:/models/ trainer_vm:latest` (automatically copies the created files inside models to local machine, otherwise they will be only inside the docker)
- `docker rm <container_id>`
- `docker run --name test_vm -e WANDB_API_KEY=<WANDB_API_KEY> europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/train_model_vm:latest`
- `docker pull europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/train_model_vm:latest`

### Conda
- `conda env create -f environment.yml`
- `conda activate DTU_ML_Ops`
- `conda remove --name DTU_ML_Ops --all`
- `make -f Makefile conda`

### GCP
- create machine: `gcloud compute instances create-with-container <name_of_instance> --container-image=<ADDRESS-OF-IMAGE-IN-ARTIFACT-REGISTRY> --project=wheel-assembly-detection --zone=europe-west1-b --machine-type=c2d-standard-4 --maintenance-policy=MIGRATE --provisioning-model=STANDARD --container-restart-policy=never --create-disk=auto-delete=yes,size=20`
- locally ssh to the instance: `gcloud compute ssh --zone "europe-west1-b" "<name_of_instance>" --project "wheel-assembly-detection"`
- authenticate to the correct server: `gcloud auth configure-docker europe-west1-docker.pkg.dev`
- pull the docker from cloud: `docker pull europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/conda_wheel_assembly_detection:30bfff9d67e13b398188608b94c44662bca1fb06`
### GCP VM
- `docker ps`: shows the docker files running on the machine
- `docker logs <CONATINER_ID>` wait until its successfully pulled (when the container is finished, I have no clue hot to reach the logs, impossible)
- `docker ps`: pulled container has new ID
- `docker exec -it CONTAINER-ID /bin/bash`: starts the docker in interactive window (only the conda_wheel_assemly_detection, the rest only train the model, upload the model and exits, maybe setting the restart policy to "never" should fix this issue)
- `sudo journalctl -u konlet-startup -n 100` (show newest 100 lines only)