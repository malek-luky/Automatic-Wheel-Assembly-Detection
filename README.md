# Car Wheel Assembly Detection 🚘

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/malek-luky/Automatic-Wheel-Assembly-Detection/main?style=flat)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/malek-luky/Automatic-Wheel-Assembly-Detection/.github%2Fworkflows%2Fbuild_conda.yml?branch=main&label=Project%20Image)
![Website](https://img.shields.io/website?url=https%3A%2F%2Fdeployed-model-service-t2tcujqlqq-ew.a.run.app%2Fdocs&up_message=online&down_message=offline&style=flat&label=model)

**Authors**: Elizaveta Isianova, Lukas Malek, Lukas Rasocha, Vratislav Besta, Weihang Li

## ⚙️ Tech Stack & Tools

![GitHub top language](https://img.shields.io/github/languages/top/malek-luky/Automatic-Wheel-Assembly-Detection?style=for-the-badge&logo=python)
![Conda](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)
![Gcloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)
![wandb](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)

## 📝 Project

This project presents an attempt to find a solution for predicting the successful assembly of tires onto wheels autonomously. Currently, the method uses purely an image based classification to predict whether the tire was assembled correctly. To enrich this, we attempt to use an LSTM model to analyze inputs from torque and force sensors of the assembling robot, enabling the system to determine the optimal conditions for tire assembly without human intervention. The goal is to increase efficiency and accuracy in tire assembly processes, reducing the reliance on manual labor and minimizing errors.

### Motivation

The project is based on real case situated within the Testbed for Industry 4.0 at CTU Prague. The current quality control methodology uses CNNs for the visual inspection of tire assemblies.

### Data

Data are meassured and labelled by the lab. The dataset is generated through robotic cell runs, every sample is then labeled as _true_ (successful assembly) or _false_ (unsuccessful assembly).

### Project Goal

This project aims to introduce a new method for enhancing the quality control process in car wheel assembly executed by a delta robot.

### Approach

Departing from the picture-based assessment using CNNs, our approach aims to evaluate the correctness of the assembly based on the data from a force-torque sensor. This transforms the dataset into a collection of time series, capturing recorded sensor data from individual tire assemblies. Each element from the series is a 6D vector combining a 3 DOF force vector and a 3 DOF torque vector.

### Methodology

The chosen methodology is an implementation of Long Short-Term Memory Recurrent Neural Networks (LSTM RNNs) using PyTorch since the data are in timeseries. There is no existing baseline solution for the current problem. Therefore the project could be evaluated and compared to the existing CNN approach.

### Limitations

Due to the small dataset limited by the time constraints and the amount of labelled data, we don't expect to obtain a well performing model, but rather want to present a method for further development.

### Framework

As a third-party framework we are going to use PyTorch Lightning and maybe with a Pytorch Forecasting package built on top of the Lightning.

## 🐍 Conda Installation

### Create the environment, install the dependencies and download the data

```
git clone https://github.com/malek-luky/Automatic-Wheel-Assembly-Detection.git
cd Automatic-Wheel-Assembly-Detection
make conda
```

## 🐳 Docker

This will build an image of our project and run it in a container. In the container you will have all the dependencies, data and code needed to run the project. We have three different dockerfiles:
- conda_setup: debugging purposes, sets the environement and waits for user to run it in interactive mode
- train_model: downloads dependencies, trains model and send it to Weight and Biases (wandb)
- deploy_model: downloads dependencies and the model from wandb and waits for user input to make predictions

The following steps to build and run are written for train_model only, but it can be easily changed for any dockerfile.

### Build the container locally after downloading the repository. 
```
git clone https://github.com/malek-luky/Automatic-Wheel-Assembly-Detection.git
cd Automatic-Wheel-Assembly-Detection
<uncomment line 21 and 22 inside dockerfiles/train_model.dockerfile>
docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest
docker run --name trainer -e WANDB_API_KEY=<WANDB_API_KEY> trainer:latest
```

### Pulls the docker image from GCP Artifact Registry
There is an error while loading the data from the bucket. Unfortunately, there is no workaround at this moment.
```
make train_model
docker run --name trainer -e WANDB_API_KEY=<WANDB_API_KEY> europe-west1-docker.pkg.dev/wheel-assembly-detection/wheel-assembly-detection-images/train_model:latest
```


## 💻 Google Cloud Computing

### Create VM Machine in GCP

1. Open [Compute Engine](https://console.cloud.google.com/compute/instances?project=wheel-assembly-detection)
2. Create a name
3. Region: `europe-west1 (Belgium)`
4. Zone: `europe-west1-b`
5. Machine configuration: `Compute-optimized`
6. Series: `C2D`
7. Machine Type: `c2d-standard-4` (must have at least 16GB RAM)
8. Boot disk: `20 GB`
9. Container image: `<ADDRESS-OF-IMAGE-IN-ARTIFACT-REGISTRY>` (click Deploy Container)
10. Restart policy: `never`
11. The rest is default

### Via gcloud command

If the `gcloud` command is unkown, [follow the steps for your OS](https://cloud.google.com/sdk/docs/install). Otherwise there are three three dockerfiles that can be deployed to Virtual Machine in GCP (suffix `_vm` to the dockerfile name`). All of the create the same instance but with specific container. The instance of the name is folowing the dockerfile name (conda_setup/train_model/deploy_model)
```
make train_model_vm
gcloud compute ssh --zone "europe-west1-b" "train-model" --project "wheel-assembly-detection"
```

### Connecting to VM machine

-   Can be via SSH inside the browser [Compute Engine](https://console.cloud.google.com/compute/instances?project=wheel-assembly-detection)
-   Or locally using command similar to this one `gcloud compute ssh --zone "europe-west1-b" "<name_of_instance>" --project "wheel-assembly-detection"` (the instatnces can be listed using `gcloud compute instances list`)

### Controlling deployed Virtual Machine

-   `docker ps`: shows the docker files running on the machine
-   `docker logs <CONATINER_ID>` wait until its successfully pulled
-   `docker ps`: pulled container has new ID
-   `docker exec -it CONTAINER-ID /bin/bash`: starts the docker in interactive window (only the conda_wheel_assemly_detection, the rest only train the model, upload the model and exits, maybe setting the restart policy to "never" should fix this issue)

## 👀 Optional

It re-creates `filtered`, `normalized` and `processed` folders. The processed data is stored in `data/processed/dataset_concatenated.csv` and is used for training.

### Re-process the data

```
python src/data/make_dataset.py
```

### Re-train the model

```
python src/models/train_model.py
```

### Run training locally without W&B
```
python src/models/train_model.py
```

### Run training locally with W&B
```
python src/models/train_model.py --wandb_on
```

### Remove the conda environment
```
conda remove --name DTU_ML_Ops --all
```

## 🌐 Deployment

This repository is configured for deployment using Google Cloud️ ☁️. The images in this repository are re-built and deployed automatically using GitHub Actions and stored in Google Artifact Registry on every push to the `main` branch.

We also automatically re-train the model using **Vertex AI**, store it in **Weights & Biases** model registry and deploy it using Google Cloud Run.

### Automatic Workflows
With access to GCP you can simply make your changes and merge it into main. When the merge is done, GitHub Actions will automatically train and deploy the model. We have 4 workflows in total.
- build_conda: build the image and stores in in GCP
- build_train: runs the built image on Vertex AI to train the model and sends it to wandb 
- build_deploy: deploy the image to cloud run to handle user requests and via FastAPI gives predictions

## 🤖 Use our model

### Cloud Deployment

The model is deployed using Google Cloud Run. You can make a prediction using the following command:

```
curl -X 'POST' \
  'https://deployed-model-service-t2tcujqlqq-ew.a.run.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": [
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2]
    ]
}'
```

### Local deployment

Our model can also be deployed locally. The guidelines for running a local server and making predictions are [here](deployment/torchserve/README.md)

## 🤝 Contributing

Contributions are always welcome! If you have any ideas or suggestions for the project, please create an issue or submit a pull request. Please follow these [conventions](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) for commit messages.

## 💻 Technology Used

-   Docker: "PC Setup" inside the docker file
-   Conda: Package manager
-   GCP
    -   Cloud Storage: Stores data for dvc pull
    -   Artifact Registry: Stores built docker images (can be created into container)
    -   Compute Engine: Enables creating virtual machines
    -   Functions / Run: Deployment
    -   Vertex AI: includes virtual machines, training of AI models ("abstraction above VM...")
-   OmegaConf: Handle the config data for train_model.py
-   CodeCov: Creates the coverage report and submit it as a comments to the pull request 
-   CookieCutter: Template used for generating code sctructure
-   DVC: Data versioning tool, similar is github but for data
-   GitHub: Versioning tool for written code
-   GitHub Actions: Run pytest, Codecov and upload built docker images to GCP
-   Pytest: Runs some tests to check whether the code is working
-   CodeCov: Tool for uploading coverage report from pytest as a comment to pull requests
-   Weight and Biases: wandb, used for storing and tracking the trained model
-   Pytorch Lightning: Framework for training our LTSM model and storing default config values
-   Forecasting: Abstracion above Pytorch Lightning working with Timeseries data
-   Torchserve: Used for local deployment
-   FastAPI: Creates API for our model, wrap it into container so it can be accessed anywhere
-   Slack/SMS: Handle the alerts, Slack for deployed model, SMS for a server cold-run

## DIAGRAM
![Diagram](reports/figures/diagram.png)

## 📂 PROJECT STRUCTURE

The directory structure of the project looks like this:

```
├── .dvc/                 <- Cache and config for data version control
├── .github/workflows     <- Includes the steps for GitHub Actions
│   └── build_conda       <- Conda dockerfile: Build conda image and push it to GCP
│   ├── build_deploy      <- Deploy dockerfile: build, push and deploy
│   └── build_train       <- Train dockerfile: Build train image and push it to GCP
│   └── pytests           <- Runs the data and model pytests
├── data                  <- Run dvc pull to see this folder
│   └── filtered          <- Seperated raw data, one file is one meassurement
│   └── normalized        <- Normalized filtered data
│   ├── processed         <- Torch sensors from normalized data and concatenated csv
│   └── raw               <- Original meassurements
├── deployment            <- Other deployment options as Cloud Function and torchserve
│   └── cloud_functions   <- File that can be run as a Cloud Function on GCP (WIP)
│   └── torchserve/       <- All data needed for local deployment
├── dockerfiles           <- Storage of out dockerfiles
│   └── conda_wheel       <- Setups the machine and open interactive environement
│   ├── train_wheel       <- Runs train_model.py that upload the new model to wandb
│   └── serve_model       <- Uses FastAPI, as the only dockerfile also deploys the model
│   └── README            <- Notes and few commands regarding the dockerfiles struggle
├── docs                  <- Documentation folder
│   ├── index.md          <- Homepage for your documentation
│   ├── mkdocs.yml        <- Configuration file for mkdocs
│   └── source/           <- Source directory for documentation files
├── reports               <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/          <- Generated graphics and figures to be used in reporting
│   └── README            <- Exam questions and project work progress
├── src                   <- Source code
│   ├── data              <- Scripts to download or generate data
│   │   └── filter        <- Seperates the meassurement into csv files
│   │   └── make_dataset  <- Runs filter->normalize->process as one script
│   │   └── normalize     <- Normalizes the filtered data
│   │   └── process       <- Changes normalized data into torch files and concatenated csv
│   │   └── README        <- Includes more details about the scripts
│   │   └── utils         <- File with custom functions
│   ├── helper            <- Folder with custom functions
│   │   └── convert_reqs  <- Function that mirrors the requirements to environment.yml
│   │   └── gcp_utils     <- Function that returns wandb_api on GCP cloud via secret
│   │   └── logger        <- Creates logs to logs/ folder for easier debugging
│   ├── models            <- Model implementations, training script and prediction script
│   │   └── arch_model    <- Old model class definition and function calls
│   │   └── arch_train_m  <- Old model using Forecasting and TemporalFusionTransformer
│   │   └── model         <- New lightweight model class definition and function calls
│   │   └── predict_model <- Predicts the result from unseen data
│   │   └── train_model   <- New lightweight model using Lightning's LTSM
├── tests                 <- Contains all pytest for Github workflow
│   └── test_data         <- Checks if data exist and the data shape
│   ├── test_model        <- Check if the trained model is correct
├── .gitignore            <- Data that are now pushed to GitHub
├── .pre-commit-config    <- Formats the code following pep8 and mirror requirements.txt
├── LICENSE               <- Open-source license info
├── Makefile              <- Makefile with convenience commands like `make data` or `make train`
├── README.md             <- The top-level README which you are reading right now
├── data.dvc              <- Links the newest data from GCP Cloud Storage
├── environment.yml       <- Requirements for new conda env, also used inside docker
├── pyproject.toml        <- Project (python) configuration file
└── requirements.txt      <- The pip requirements file for reproducing the environment
```

## 🙏 Acknowledgements

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
