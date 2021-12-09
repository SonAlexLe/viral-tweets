from flask import Flask, request, render_template
from training import trigger_retrain, complete_retrain_job
from prediction import run_model, get_last_accuracy, get_last_mlflow_id
import json

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/new-data-callback", methods=["GET", "POST"]) # either GET or POST will do, no difference
def new_data_callback():
    trigger_retrain()
    return ("ok", {"Content-Type": "text/plain; charset=utf-8"}) # (response, headers) tuple

@app.route("/retrain-complete", methods=["POST"])
def retrain_complete():
    job_id = request.form["job_id"]
    complete_retrain_job(job_id, {"accuracy": float(request.form["accuracy"]), "mlflow_run_id": request.form["mlflow_run_id"]})
    return ("ok", {"Content-Type": "text/plain; charset=utf-8"})

@app.route("/predict", methods=["POST"])
def predict():
    tweet_text = request.form["tweet_text"]
    extra_data = {
        "is_quote_tweet": request.form["is_quote"] == "1",
        "photos_count": min(4, max(0, int(request.form["photos_count"]) - 1)), # If incoming value is 0, make it 0 photos. 1 also means 0 photos, 2 means 1 photo, 3 means 2 photos and so on, up to 4 photos maximum.
        "has_video": request.form["has_video"] == "1",
    }
    
    model_input_data, predicted_classes = run_model(tweet_text, **extra_data)
    json_pretty = json.dumps(model_input_data, indent=2, sort_keys=True)
    
    return render_template("prediction.html", json_input=json_pretty, predicted_classes=predicted_classes.numpy()[0])

@app.route("/", methods=["GET"])
def index_page():
    return render_template("index.html", curr_accuracy=get_last_accuracy(), mlflow_id=get_last_mlflow_id())
