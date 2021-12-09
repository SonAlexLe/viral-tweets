# Pipeline implementation
The source code of the pipeline is split into two parts, which are meant to be set up on two separate machines.

## Retraining machine
This part of the pipeline handles retraining. The core files of this part of the pipeline are `retrain_server.py` and `pytorch_models.py`.
- `pytorch_models.py` - implements the model and its training process. This script isn't usually executed directly, instead it is called by the retraining server, described below.
- `retrain_server.py` - implements the HTTP server that handles retrain requests and uploads results to MLflow. This server won't be accessible from the outside world, but it must be reachable from the server machine.
- `constants.py` - stores constant values for easy configuration: the interesting ones are:
  - `RESULTS_SERVER_URL` - address of the server machine's main web server. Once retraining is complete, a request to this URL is made to notify of the completion.
  - `MLFLOW_SERVER_URL` - address of the MLflow server.
- `utils_data.py` - helper functions for training the model and pre-processing data.

The script `retrain_server.py` is a Flask application. One should use Flask (or any other WSGI-compatible backends like gunicorn) to run the server. The pipeline is configured as if the server were always running on the endpoint 192.168.1.227:23500 - you will need to adjust the scripts in the server machine if this is not the case.

## Server machine
This part of the pipeline is the main web server - the one that handles incoming requests from users, executes prediction on the currently deployed model and hosts MLflow. The `mlflow_service` directory contains a script for running the MLflow server; one might change this script to adjust the artifact directory, which might require a few gigabytes of free disk space.

The `http_service` directory contains the main web server. The core file is `app.py` - again, a Flask application. By default, it's expected to be listening on 192.168.1.223:5001 - if this is not the case, you will need to adjust the address in the retraining machine's `constants.py` file. Other files include:
- `mlflow_binding.py` - helper functions for facilitating communication to and from the MLflow server.
- `accuracy_testing.py` - a function for testing the previous model on new data, used to decide whether a new model should be deployed.
- `prediction.py` - everything related to performing a single prediction on a Tweet text. It exposes the currently deployed model as a set of convenient functions.
- `training.py` - everything related to retraining. This script contains the implementation of the retraining thread - communication with the retraining machine and model redeployment.
- `config.py` - configuration data: the interesting entries are:
  - `MLFLOW_SERVER_URL` - address of the MLflow server.
  - `TRIGGER_TRAINING_URL` - address of the retraining machine's HTTP server.
