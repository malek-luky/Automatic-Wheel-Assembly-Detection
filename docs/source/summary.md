# Project Summary
We successfuly deployed the model, while followign all the exercises and instructions on the course page. The framework we used is already mentioned in the report as well as in the README file, therfore here we would liek to say few words about what we are proud about and that was not really mandatory for the scope of the project as well as some tasks that were not very relevant for our project.

## Irrelevant Features

### Data drifting
The data drigting happens because the new data can not follow the distribution it was trained on or the world simply just change. Our use is to check whether the tyre and wheel was correctly assembled. In this scenario, data drifting is most likely not gonna happen and in case we start assembling new tyre, we must reatrain the entire model.

### Prunning
Pruinning should speed up the predicitons, removing the synapses that are not really relevant for the final prediction. However, our model is too simple, so the prediction does not take much time at all. For more complex models in the future, it might be relevant, but for no there is no reason for implementing it.

### Distributed data loading and model training
As mentinoed earlier, the model is rather simple, therefore there is not need to distribute the model training. The same is true for the data. We are working with a very small dataset, counting only 100 meassurements. So distributing the data loading does not amke any sense in our case.

### Saving Checkpoints
It's a good practice to also store checkpoints while training the model, to make sure that we can easily continue our training and we do not loose entire progress in case of an error. However for our use case, since the model is very small, this feature is not relevant.

## What are we proud of 

### Mkdocs
We made quite handy MkDocs for our project. ALthough it can be further improved, it gives somethign extra for the project and might be useful for others as well as for us if we come back to this repository in our future.

### Alerts
We implemented alerts for the Slack group that we are using to communciate, we receive a notification everytime when new model is deployed, that allows us to e.g. check whether the new model is giving reasonable results or just make sure that the deployment was correct. We are also getting SMS notification everytime when the cloud run is triggered.

### Diagram
The diagram that shows all the frameworks we used in our project is quite easy to orient in and graphicaly summarizes what tools we used during this project.

### Automated pep8 formating
As one of the precommits we implemented automatic code formatting. Therefore we can be sure that the code is following the standard even if the commit is not perfect.

### Coverage Report
Our coverage report for each pytest is automatically uplloaded as a comment to each pull request. Therefore it is quite easy to see what coverage we achieve during the pytests.

### Hyperparameter tuning
By running the `src/models/traun_model.py` with argument `-sweep`, we automcatically do the hyperparameter optimization and store it to wandb for future analysis.