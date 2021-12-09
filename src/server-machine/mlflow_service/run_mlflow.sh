#!/bin/bash
screen -d -m -S mlflow_tracking_server mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///~/mlflow_service/mlflow_data.db --default-artifact-root file:/srv/mlflow_nfs/
