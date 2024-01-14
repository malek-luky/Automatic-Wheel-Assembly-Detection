## 📄Generating the docs

- Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation. 
- Build locally with:  `mkdocs build`
- Serve locally with: `mkdocs serve`

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

**(Optional): Remove the conda environment**

```
conda remove --name DTU_ML_Ops --all
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