# Pipeline implementation
The source code of the pipeline is split into two parts, which are meant to be set up on two separate machines.

## Retrain machine
This part of the pipeline handles retraining. The core files of this part of the pipeline are `retrain_server.py` and `pytorch_models.py`.
- `pytorch_models.py` - implements the model and its training process. This script isn't usually executed directly, instead it is called by the retrain server, described below.
- `retrain_server.py` - implements the HTTP server that handles retrain requests and uploads results to MLflow. This server won't be accessible from the outside world, but it must be reachable from the server machine.
- `constants.py` - stores constant values for easy configuration: the interesting ones are:
  - `RESULTS_SERVER_URL` - the address of the server machine's main web server. Once retraining is complete, a request to this URL is made to notify of the completion.
  - `MLFLOW_SERVER_URL` - the address of the MLflow server.
- `utils_data.py` - helper functions for training the model and pre-processing data.

The script `retrain_server.py` is a Flask application. One should use Flask (or any other WSGI-compatible backends like gunicorn) to run the server. The pipeline is configured as if the server were always running on the endpoint 192.168.1.227:23500 - you will need to adjust the scripts in the server machine if this is not the case.
## Server machine
