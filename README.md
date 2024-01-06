# Wheel_Assembly_Detection

## TODO
* [ ] Add data to data/raw
* [ ] Add normalized and processed data to data/processed
* [ ] Create model and put it into models
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

## PROJECT DESCRIPTION

TL;DR: A project with no prior study that  to predict a successful assembly of tires on wheels without the need for operator intervention using data from torque and force sensor.

### MOTIVATION
The project is based on real case situated within the Testbed for Industry 4.0 at CTU Prague. The current quality control methodology uses CNNs for the visual inspection of tire assemblies.

### DATA
Data are meassured and labelled by the lab. The dataset is generated through robotic cell runs, every sample is then labeled as ‘true’ (successful assembly) or ‘false’ (unsuccessful assembly). We are using a “smaller” dataset (size limited due to self-labelling the data).

### PROJECT GOAL
This project aims to introduce a new method for enhancing the quality control process in car wheel assembly executed by a delta robot.

### APPROACH
Departing from the picture-based assessment using CNNs, our approach aims to evaluate the correctness of the assembly based on the data from a force-torque sensor. This transforms the dataset into a collection of time series, capturing recorded sensor data from individual tire assemblies. Each element from the series is a 6D vector combining a 3 DOF force vector and a 3 DOF torque vector.

### METHODOLOGY
The chosen methodology is an implementation of Long Short-Term Memory Recurrent Neural Networks (LSTM RNNs) using PyTorch since the data are in timeseries. There is no existing baseline solution for the current problem, so some part of the project time would be spent on experimenting with models.

### FRAMEWORK
As a third-party framework we are going to use PyTorch Lightning and maybe with a Pytorch Forecasting package built on top of the Lightning.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── Wheel_Assembly_Detection  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
