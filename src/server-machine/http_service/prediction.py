from threading import Lock
from transformers import AutoTokenizer
import pandas as pd
import config
import torch
import datetime
import re
import os
import json

class InferenceModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2", normalization=True)
        self.current_model = None
        self.mlflow_run_id = None
        self.model_accuracy = None # only for potential statistics/debugging purposes, not actually used in any comparisons
        self.reload_mutex = Lock()
        self.reload_immediately()

    def reload_immediately(self): # call this to reload the model immediately
        self.reload_mutex.acquire() # thread-safety in case a multithreaded webserver is used
        try:
            if os.path.exists(config.MODEL_FILENAME):
                self.current_model = torch.jit.load(config.MODEL_FILENAME)
                
            try:
                with open(config.MODEL_METADATA_FILENAME, "r", encoding="utf-8") as file:
                    model_metadata = json.load(file)
                    self.mlflow_run_id = model_metadata["mlflow_run_id"]
                    self.model_accuracy = model_metadata["model_accuracy"]
            except:
               pass # ignore any errors that could have occurred: when the model is retrained, the metadata will be updated anyway

        finally:
            self.reload_mutex.release()

    def get_model(self):
        self.reload_mutex.acquire()
        try:
            return self.current_model
        finally:
            self.reload_mutex.release()

inf_model = InferenceModel()

def reload_model(new_accuracy, new_mlflow_run_id):
    try:
        with open(config.MODEL_METADATA_FILENAME, "w", encoding="utf-8") as file:
            json.dump({"model_accuracy": new_accuracy, "mlflow_run_id": new_mlflow_run_id}, file)
    except:
        pass # ignore filesystem errors

    inf_model.reload_immediately()

def get_model():
    return inf_model.get_model()

def get_last_accuracy():
    return inf_model.model_accuracy

def get_last_mlflow_id():
    return inf_model.mlflow_run_id

def count_hashtags(text):
    return len(re.findall("#[^ ]+", text))

def count_mentions(text):
    return len(re.findall("@[^ ]+", text))

def count_links(text):
    return len(re.findall("https{0,1}:\/\/", text))

# tweet_text is a string containing the tweet text
def construct_input_tensor(tweet_text, full_data_entry = None, is_quote_tweet = False, photos_count = 0, has_video = False):
    datenow = datetime.datetime.utcnow()

    d = {
        "created_at_dayofweek": datenow.weekday(),
        "created_at_hour": datenow.hour,
        "hashtags_count": count_hashtags(tweet_text),
        "mentions_count": count_mentions(tweet_text),
        "is_quote_tweet": is_quote_tweet,
        "word_count": len(tweet_text.split()),
        "photos_count": photos_count,
        "has_video": has_video,
        "urls_count": count_links(tweet_text)
    }
    if full_data_entry is not None: # actually disregard the previous data, just keep the columns
        d = {k: full_data_entry[k] for k in d.keys()}
    
    df = pd.DataFrame({k: [v] for k, v in d.items()})

    item = inf_model.tokenizer(tweet_text, return_tensors="pt")
    item["features"] = torch.tensor(df[list(d.keys())].values.astype("float32"))

    d["tweet"] = tweet_text # only for display purposes
    return d, item

def run_model(tweet_text, **kwargs):
    model = inf_model.get_model()
    full_data, tensors = construct_input_tensor(tweet_text, **kwargs) # augment the tweet text with extra metadata used as features
    predicted_class, predicted_class_prob = model(*tensors.values()) # run prediction
    
    return (full_data, predicted_class_prob) # full_data is the input dictionary to the model (before converting to tensors), predicted_class_prob is the model's result
