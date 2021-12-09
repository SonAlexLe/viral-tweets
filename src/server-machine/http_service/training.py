from urllib.request import Request, urlopen
from threading import Thread, Event

from prediction import get_model, reload_model
from accuracy_testing import should_replace_model

import mlflow_binding
import logging
import os, tempfile, urllib
import secrets

import config

pending_retrains = {} # a dictionary of TrainingJob: key is job_id, value is the TrainingJob object (see below)

class TrainingJob():
    def __init__(self):
        self.event = Event()
        self.data = None

def complete_retrain_job(job_id, data):
    try:
        pending_retrains[job_id].data = data
        pending_retrains[job_id].event.set()
    except KeyError:
        pass # ignore a situation where we complete a non-existent job, it does no harm

def get_logger():
    logger = logging.getLogger("retrain_thread")
    if logger.handlers: # if logger has handlers, it has already been initialized and we don't need to do that again
        return logger

    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # log to file
    file_handler = logging.FileHandler(filename="last_retrain.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def make_retrain_request(data_url, job_id):
    """Returns True if request succeeded, otherwise returns False."""
    
    train_req_post_fields = {
        "data_url": data_url,
        "training_job_id": job_id
    }
    
    try:
        train_request = Request(config.TRIGGER_TRAINING_URL, data=urllib.parse.urlencode(train_req_post_fields).encode(), method="POST")
        return urlopen(train_request, timeout=5).read().decode("utf-8") == "ok"
    except Exception as err:
        get_logger().exception(err)
    
    return False

def retrain_thread():
    get_logger().info("Started retraining monitoring thread. Parameters:")
    
    data_url = config.get_tweets_data_url()
    job_id = secrets.token_hex(16)
    
    get_logger().info(f" > Tweets data source: {data_url}")
    get_logger().info(f" > job ID: {job_id}")
    get_logger().info(f" > endpoint for triggering training: {config.TRIGGER_TRAINING_URL}")
    
    if not make_retrain_request(data_url, job_id):
        get_logger().error("Retrain request failed. Perhaps the other server is not responding?")
        return

    get_logger().info("Retrain request sent successfully. Waiting for response...")
    training_job = TrainingJob()
    
    try:
        pending_retrains[job_id] = training_job
        if not training_job.event.wait(timeout=7200): # wait for 2 hours
            get_logger().error("Timed out while waiting for retraining result. Job ignored.")
            return
    finally:
        del pending_retrains[job_id]
    
    new_accuracy = training_job.data["accuracy"]
    new_mlflow_run_id = training_job.data["mlflow_run_id"]
    get_logger().info(f"Retrain result: MLflow run ID {new_mlflow_run_id}, new accuracy {new_accuracy:.3f}")
    get_logger().info("Checking if the model should be replaced...")
    
    if should_replace_model(new_accuracy, new_mlflow_run_id, get_model()):
        get_logger().info("The current model will be replaced - downloading from MLflow server...")
        
        mlflow_binding.download_model(training_job.data["mlflow_run_id"], config.MODEL_FILENAME)
        reload_model(new_accuracy, new_mlflow_run_id)
        
        get_logger().info("New model installed successfully!")
    else:
        get_logger().info("The current model will NOT be replaced.")
    
    get_logger().info("Retraining thread completed, exiting")
    

def trigger_retrain():
    Thread(target=retrain_thread).start()
