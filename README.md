# 🏡 House Price Prediction Using MLflow

## 📌 Overview
This project trains a machine learning model to predict house prices using the **California Housing Dataset**. It also logs hyperparameters and performance metrics using **MLflow**.

## 🚀 Features
- **Hyperparameter Tuning** with `GridSearchCV`
- **MLflow Tracking** for experiments and model comparisons
- **Best Model Selection** and registration

## 📁 Project Structure
- `src/data_preparation.py` → Loads dataset.
- `src/train.py` → Trains model & performs hyperparameter tuning.
- `src/mlflow_tracking.py` → Logs experiments.


## 🛠️ Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/house-price-prediction-mlflow.git
cd house-price-prediction-mlflow

# Create virtual environment
conda create -p venv python=3.10
conda activate venv/

# Install dependencies
pip install -r requirements.txt