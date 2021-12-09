import urllib.request
import re

TRIGGER_TRAINING_URL = "http://192.168.1.227:23500/retrain"
MLFLOW_SERVER_URL = "http://192.168.1.223:5000/"

MLFLOW_MODEL_FILENAME = "model.pt" # the name of the model artifact on the MLflow server
MODEL_FILENAME = "current_model.pt" # local file name of the currently used model
MODEL_METADATA_FILENAME = "current_model.json" # local file name of the currently used model's metadata 

def get_tweets_data_url():
    base_url = "https://users.aalto.fi/~higginm1/tweets_project/by_day/"
    contents = urllib.request.urlopen(f"{base_url}?C=M;O=D;F=0").read().decode("utf-8")
    
    parent_dir_index = contents.index("Parent Directory")
    match = re.search("a href=\"(.*)\"", contents[parent_dir_index:])
    
    return f"{base_url}{match[1]}"
    