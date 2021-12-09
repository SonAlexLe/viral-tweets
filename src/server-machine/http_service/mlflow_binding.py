import config
import os
from mlflow.tracking import MlflowClient

def download_model(run_id, target_path):
    client = MlflowClient(tracking_uri=config.MLFLOW_SERVER_URL)
    client.download_artifacts(run_id, "model", ".")
    os.replace("model/saved_model.pt", target_path)
    os.rmdir("model") # the model directory should now be empty as we have moved its only file out of it

def download_test_data(run_id):
    client = MlflowClient(tracking_uri=config.MLFLOW_SERVER_URL)
    run_params = client.get_run(run_id).data.params
    
    data_dir = client.download_artifacts(run_id, "data")
    return os.path.join(data_dir, run_params["test_data_path"])
