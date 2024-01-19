## Improvements
On this page we would like to mention possible improvements. 

### Make more complex model
We did not have enoguh data to make a complex model. But adding more hidden layers or multiple LTSM sequences next to each other could potentially improve the results.

### Try locust
Would be great to see how GCP behaves when multiple requests are happening at the same time. For that we could use the Locuts python package. to try it out

### Experiment with Forecasting
In the beginning of the course, we used forecasting package to use for our model. Nevertheless, due to the fact that the package was outdated and also quite complex, we did not use it in the end. Would be nice to experiement a bit more with it and see what does it.

### Upload the best model to model registry
Inside the training, we are uploading the new model to model registry. But the point of model registry is to use only the best model we have. Therefore it would be great to implement some logic which would compare the newly trained model and the one we have online to see which one gives better results on a validation test and keep that one as the deployed model used for users predictions.

### Add caching
The caching, even after spending quite a lot of tiem with it still does not work. It does not speed up anything but instead takes quite a lot of space on GitHub. For that reason we decided not to use cache in our project.

### Mirroirng requirements (WIP)
We got tried of copying all the requirements to conda environemnt file as well. Therefore we made another pre-ommit that should automatically take alle new requirements from the requirements.txt file and copy them to environemnt.yml for conda.