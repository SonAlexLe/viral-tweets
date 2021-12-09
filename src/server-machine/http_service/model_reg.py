import mlflow
import config
from mlflow.tracking import MlflowClient

# delete this file if deemed unnecessary

def get_model_info(model_name, stage):
    client = MlflowClient(tracking_uri=config.MLFLOW_SERVER_URL)
    mv = client.get_latest_versions(model_name, stages=[stage])
    assert len(mv) == 1, "Only 1 model in any stage at a time!"
    # return client.get_model_version_download_uri(movel_name, mv[0].version)
    return mv[0]


def load_model(model_name, stage):
    download_uri = get_model_info(model_name, stage).source
    return mlflow.pyfunc.load_model(download_uri)  # a tokenizer is already packaged with this object



