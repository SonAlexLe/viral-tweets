from prediction import get_last_mlflow_id, construct_input_tensor
import mlflow_binding
import pandas as pd

def should_replace_model(new_accuracy, new_mlflow_id, previous_model):
    if get_last_mlflow_id() is None or previous_model is None:
        return True # if the current model stats are unknown, replace just to be sure
    
    test_data_path = mlflow_binding.download_test_data(new_mlflow_id)
    df = pd.read_json(test_data_path, lines=True)
    
    def run_model(row):
        _, tensors = construct_input_tensor(row["tweet"], row.to_dict())
        pred, _ = previous_model(*tensors.values())
        return pred.squeeze().item()
    
    y_true = df.label.to_numpy()
    y_pred = df.apply(run_model, axis=1)
    old_accuracy = (y_true == y_pred).mean()
    print(f"Old model on new data: {old_accuracy}")
    
    return old_accuracy < new_accuracy
