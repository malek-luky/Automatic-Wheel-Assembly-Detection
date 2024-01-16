.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = DTU_ML_Ops
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

 ## Create Conda environment (pip install -e . download local packages)
conda:
	pip install -e . 
	conda env create -f environment.yml
	conda activate DTU_ML_Ops
	dvc pull

## Docker Online
docker_conda_online:
	gcloud auth configure-docker europe-west1-docker.pkg.dev
	docker pull europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/conda_setup:latest

docker_train_online:
	gcloud auth configure-docker europe-west1-docker.pkg.dev
	docker pull europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/train_model:latest

docker_deploy_online:
	gcloud auth configure-docker europe-west1-docker.pkg.dev

## Create VM Machine
virtual_machine:
	gcloud compute instances create-with-container instance_makefile --container-image=<ADDRESS-OF-IMAGE-IN-ARTIFACT-REGISTRY> --project=wheel-assembly-detection --zone=europe-west1-b --machine-type=c2d-standard-4 --maintenance-policy=MIGRATE --provisioning-model=STANDARD --container-restart-policy=never --create-disk=auto-delete=yes,size=20
	gcloud compute ssh --zone "europe-west1-b" "<name_of_instance>" --project "wheel-assembly-detection"

## Process raw data into processed data
data:
	dvc pull
	python $(PROJECT_NAME)/data/make_dataset.py

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')