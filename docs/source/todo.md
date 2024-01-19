# Slack Canva
Here we would like to store our Slack Canva that we used for distributing the tasks throughout the course.

## Week 1
-   [x] Create a git repository (Lukas M.)
-   [x] Make sure that all team members have write access to the github repository (Lukas M.)
-   [x] Create a dedicated environment for you project to keep track of your packages (Everyone)
-   [x] Create the initial file structure using cookiecutter (Lukas M.)
-   [x] Fill out the make_dataset.py file such that it downloads whatever data you need and (Vraťa)
-   [x] Add a model file and a training script and get that running (Lukas R. +  Liza + Weihang)
-   [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code (Lukas R. + Weihang)
-   [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. (Lukas R.)
-   [x] Used Hydra to load the configurations and manage your hyperparameters nope, will be replaced by Lightning CLI 
-   [x] Remember to fill out the requirements.txt file with whatever dependencies that you are using (Lukas M.)
-   [x] Setup version control for your data or part of your data (Lukas R.)
-   [x] Construct one or multiple docker files for your code (Lukas M.)
-   [x] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
-   [x] Build the docker files locally and make sure they work as intended 

## Week 2

-   [x] Write unit tests related to the data part of your code (Lukas M.)
-   [x] Update cache command (Lukas M.)
-   [x] Write unit tests related to model construction and or model training (the tests should be written into test folder and the workflow into .github/workflow folder) (liza)
-   [x] Read the csv from datafolder (Lukas M.)
-   [x] Consider running a hyperparameter optimization sweep. (Liza) 
-   [x] Update the dvc bucket with the new files (Lukas R.)
-   [x] path while running `python src/data/make_dataset.py` is wrong inside conda (works for windows/ubuntu tho, would be nice if someone can double check)
-   [x] columns for `src/model/train_model.py` are not the same as inside data/processed/dataset_concatenated.csv`
-   [x] Calculate the coverage.
-   [x] Get some continuous integration running on the github repository (Lukas M.)
-   [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup (Lukas R.)
-   [x] Create a trigger workflow for automatically building your docker images (Lukas R.)
-   [x] Get your model training in GCP using either the Engine or Vertex AI
-   [x] Create a FastAPI application that can do inference using your model (Lukas R.)
-   [x] If applicable, consider deploying the model locally using torchserve (liza)
-   [x] Deploy your model in GCP using either Functions or Run as the backend
-   [x] Wandb monitoring  (Lukas R)  https://wandb.ai/02476mlops/automatic-wheel-assembly-detection?workspace=user-lukyrasocha
-   [x] Figure out wandb auth stuff so you can also monitor runs when training via docker (Lukas R)
-   [x] LOGGING!!!! (Lukas R)
-   [x] Save trained model locally (Lukas R.)
-   [x] Save trained model in cloud (so that we can access models that were trained in cloud) (Lukas M.)
-   [x] Hyperparameters (now they are set in the beginning of the file, try calling the training via the client and not the file itself... Try using Lightning CLI? or hydra? → OmegaConf (Liza)
-   [x] Try automatic hyperparameter tuning using optuna/Lighngtning CLI/Forecasting → WandB (Liza)

## Week 3

-   [x] Create documentation using MkDocs (include there your personal notes or the readme from the docs folder) (Lukas M.)
-   [x] Answer the questions that are part of the report
-   [x] Setup monitoring for the system telemetry of your deployed model?
-   [x] Setup monitoring for the performance of your deployed model?



## BRAINSTORM

-   [x] Do we want so save and load checkpoints as well? (might be good practice for large-scale models)
-   [x] Do we want to somehow optimize the parameter tuning (e.g. start already tuning around the best parameters from previous runs maybe?)
-   [x] We are uploading and using last trained model, should we use the best model instead?
-   [x] Should we have just environment.yaml and remove requirements.txt? To reduce overhead



## CHECK BEFORE SUBMISSION

-   [x] Remember to comply with good coding practices (pep8) while doing the project
-   [x] Do a bit of code typing and remember to document essential parts of your code
-   [x] Check whether docker runs correctly if started from scratch
-   [x] Save slack canva to README
-   [x] Update the fodler structure in README
-   [x] Try making new conda environment and fill all missing/wrong requirements
-   [x] Add branch protection rules to check all pytest before merging
-   [x] Delete useless data from GCP bucket
-   [x] Revisit your initial project description. Did the project turn out as you wanted?
-   [x] Make sure all group members have a understanding about all parts of the project
-   [x] Check if all your code is uploaded to github
-   [x] Change default flag train to sweep? (now its only -wandb_on)
-   [x] Check Coverage Report (now it does not work)(Lukas M.)



## UNNECESSARY

-   [x] If applicable, play around with distributed data loading
-   [x] If applicable, play around with distributed model training
-   [x] Check how robust your model is towards data drifting
-   [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed

