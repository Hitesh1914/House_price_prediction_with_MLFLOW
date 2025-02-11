import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature
from urllib.parse import urlparse
from data_preparation import load_data
from mlflow_tracking import log_experiment, setup_mlflow_tracking, log_model
data = load_data()

# Split features & target
X = data.drop(columns=["Price"])
y = data["Price"]

## Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

## Hyperparameter tuning using Grid Searchcv
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search

# Start MLflow experiment

with mlflow.start_run():
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid) ##perform hyperparametet tunning
    best_model = grid_search.best_estimator_     ## Get the best model

    y_pred = best_model.predict(X_test)            ## Evaluate the best model
    mse = mean_squared_error(y_test, y_pred)

    log_experiment(best_model, grid_search, mse)  ## Log the best parameters and metrics 

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Mean squared error: {mse}")
