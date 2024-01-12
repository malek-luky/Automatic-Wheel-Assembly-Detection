# Car Wheel Assembly Detection 🚘

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/malek-luky/Automatic-Wheel-Assembly-Detection/main?style=for-the-badge)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/malek-luky/Automatic-Wheel-Assembly-Detection/.github%2Fworkflows%2Fconda-build.yml?branch=main&style=for-the-badge&label=Project%20Image%20Build)
![Website](https://img.shields.io/website?url=http%3A%2F%2Fmodel-deploy.dk%2F&up_message=online&down_message=offline&style=for-the-badge&label=model)

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

## 🚀 Getting Started

Steps to build the repository in conda or docker

### 🐍 Conda

**Clone the repository**

```
git clone https://github.com/malek-luky/Automatic-Wheel-Assembly-Detection.git

cd Automatic-Wheel-Assembly-Detection
```

**Create the environment and install the dependencies**

```
conda env create -f environment.yml

conda activate DTU_ML_Ops
```

**Download the data**

This will download the data from a GCP bucket using DVC and place it in `data` folder.

```
dvc pull
```

**(Optional): Re-process the data from `data/raw`. It re-creates `filtered`, `normalized` and `processed` folders. The processed data is stored in `data/processed/dataset_concatenated.csv` and is used for training.**

```
python src/data/make_dataset.py
```

**(Optional): Re-train the model**

```
python src/models/train_model.py
```

### 🐳 Docker

This will build an image of our project and run it in a container. In the container you will have all the dependencies, data and code needed to run the project.

**Clone the repository**

```
git clone https://github.com/malek-luky/Automatic-Wheel-Assembly-Detection.git

cd Automatic-Wheel-Assembly-Detection
```

**Build the image**

```
docker build -f dockerfiles/conda_wheel_assembly_detection.dockerfile . -t wheel:latest
```

**Run the container**

```
docker run --name wheel_container -it --entrypoint /bin/bash wheel:latest
```

## 🌐 Deployment

This repository is configured for deployment using Google Cloud️ ☁️. The images in this repository are re-built and deployed automatically using GitHub Actions and stored in Google Artifact Registry on every push to the `main` branch.

When there are changes to the model or the data, we also automatically re-train the model store it in Google Cloud Storage and deploy it using Google Cloud Run.

## 🤖 Use our model

The model is deployed using Google Cloud Run. You can make a prediction using the following command:

```
curl -X POST https://INSERT-OUR-URL/predict -H "Content-Type: application/json" -d '{"data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}'
```

## 🤝 Contributing

Contributions are always welcome! If you have any ideas or suggestions for the project, please create an issue or submit a pull request. Please follow these [conventions](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) for commit messages.

## 🙏 Acknowledgements

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
