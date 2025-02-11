import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

def log_experiment(model, grid_search, mse):
    #Logs the best paramenters and metrics to MLflow
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(model, "model")

def setup_mlflow_tracking():
    ##Set up the MLflow tracking server

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

def log_model(best_model, signature=None, model_name="Best RandomForest Model"):

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # scheme is http

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name)
    else:
        mlflow.sklearn.log_model(best_model, "model", signature=signature)