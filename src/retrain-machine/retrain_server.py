from flask import Flask, request
from threading import Thread
from importlib import reload
from urllib.request import Request, urlopen
import pytorch_models, utils_data, constants
import logging
import urllib
import mlflow
import os

# disable the https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
# warning that would occur if the server is still active after the first GET
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

def retrain_thread(data_url, training_job_id):
    mlflow.set_tracking_uri(constants.MLFLOW_SERVER_URL)
    with mlflow.start_run() as run:
        app.logger.info(f"Starting MLflow run ID {run.info.run_id}")

        processed_data_path = utils_data.preprocess_pipeline(data_url, app.logger, use_temp=False)
        # processed_data_path = "tweets_covid.json.xz"

        app.logger.info(f"Running training. This make take some time...")
        training_result = pytorch_models.run_training(processed_data_path)

        app.logger.info(f"Submitting results into MLflow...")
        mlflow.log_metrics(training_result["results"])
        mlflow.log_params(training_result["config"])
        # mlflow.pyfunc.log_model(
        #     "model",
        #     loader_module="mlflow_model",
        #     data_path="saved_model.pt",
        #     code_path=["mlflow_model.py"],
        #     signature=mlflow_model.get_model_signature(),
        #     conda_env=mlflow_model.get_conda_yaml_dict(),
        #     registered_model_name="TweetClassifier"
        # )
        mlflow.log_artifact("saved_model.pt", artifact_path="model")
        mlflow.log_artifact(processed_data_path, artifact_path="data")
        mlflow.log_artifact(training_result["config"]["test_data_path"], artifact_path="data")

        os.unlink(training_result["config"]["test_data_path"])
        os.unlink(processed_data_path)
        
        app.logger.info(f"Finally, notifying the remote server of success...")
        result_post_fields = {
            "job_id": training_job_id,
            "accuracy": training_result["results"]["accuracy"],
            "mlflow_run_id": run.info.run_id
        }
        
        result_request = Request(
            constants.RESULTS_SERVER_URL,
            data=urllib.parse.urlencode(result_post_fields).encode(),
            method="POST"
        )
        urlopen(result_request, timeout=20)
        
        app.logger.info(f"Retraining completed.")

@app.route("/retrain", methods=["POST"])
def predict():
    data_url = request.form["data_url"]
    training_job_id = request.form["training_job_id"]
    Thread(target=retrain_thread, args=(data_url, training_job_id)).start()
    return "ok"

@app.route("/", methods=["GET"])
def main_page():
    return ("This is a server that handles model retraining.", {"Content-Type": "text/plain; charset=utf-8"})
