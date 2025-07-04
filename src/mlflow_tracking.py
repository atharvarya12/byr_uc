import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import json
import pickle

# === Load metadata ===
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

# === Log Random Forest ===
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with mlflow.start_run(run_name="RandomForest"):
    mlflow.log_params(metadata["rf"]["params"])
    mlflow.log_metric("roc_auc", metadata["rf"]["roc_auc"])
    mlflow.log_metric("f1_score", metadata["rf"]["f1_score"])
    mlflow.sklearn.log_model(rf_model, "model")

    mlflow.log_artifact("reports/RandomForest_classification_report.txt")
    mlflow.log_artifact("reports/RandomForest_roc_curve.png")

print("✅ Logged Random Forest to MLflow")

# === Log XGBoost ===
with open("models/xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with mlflow.start_run(run_name="XGBoost"):
    mlflow.log_params(metadata["xgb"]["params"])
    mlflow.log_metric("roc_auc", metadata["xgb"]["roc_auc"])
    mlflow.log_metric("f1_score", metadata["xgb"]["f1_score"])
    mlflow.xgboost.log_model(xgb_model, "model")

    mlflow.log_artifact("reports/XGBoost_classification_report.txt")
    mlflow.log_artifact("reports/XGBoost_roc_curve.png")

print("✅ Logged XGBoost to MLflow")
