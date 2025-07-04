print("🚀 Script started...")

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from utils import save_pickle, compress_model

# === Create folders ===
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# === Load Data ===
DATA_PATH = os.path.join("data", "ProcessedData.csv")
df = pd.read_csv(DATA_PATH)
df = shuffle(df, random_state=42)

X = df.drop(columns=["final_status_success", "Unnamed: 0"])
# Save encoded column names
encoded_X = pd.get_dummies(X, drop_first=True, dtype=int)
with open("models/encoded_columns.pkl", "wb") as f:
    pickle.dump(encoded_X.columns.tolist(), f)

y = df["final_status_success"]


# === Split and apply SMOTE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("🧪 Before SMOTE:", y_train.value_counts().to_dict())
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("✅ After SMOTE:", y_train.value_counts().to_dict())

# === Utility: Save classification report
def save_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred)
    print(f"📊 Classification Report - {model_name}:\n{report}")
    path = os.path.join("reports", f"{model_name}_classification_report.txt")
    with open(path, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved: {path}")

# === Utility: Plot and Save ROC Curve ===
def plot_roc(y_true, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc='lower right')
    plot_path = os.path.join("reports", f"{model_name}_roc_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved ROC curve for {model_name} at: {plot_path}")
    return roc_auc

# === Random Forest + GridSearch ===
print("🔍 Running GridSearchCV for Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='f1', verbose=1, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
save_classification_report(y_test, y_pred_rf, "RandomForest")
rf_probs = best_rf.predict_proba(X_test)[:, 1]
roc_auc_rf = plot_roc(y_test, rf_probs, "RandomForest")
print(f"🎯 AUC (Random Forest): {roc_auc_rf:.4f}")
save_pickle(best_rf, "models/random_forest.pkl")
compress_model("models/random_forest.pkl", "models/random_forest.tar.gz")

# === XGBoost + RandomizedSearch ===
print("🔍 Running RandomizedSearchCV for XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [0.5, 1, 2, 5]  # Helps with class imbalance
}
xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_rand = RandomizedSearchCV(xgb_base, xgb_params, n_iter=10, scoring='f1', cv=3, verbose=1, n_jobs=-1, random_state=42)
xgb_rand.fit(X_train, y_train)
best_xgb = xgb_rand.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
save_classification_report(y_test, y_pred_xgb, "XGBoost")
xgb_probs = best_xgb.predict_proba(X_test)[:, 1]
roc_auc_xgb = plot_roc(y_test, xgb_probs, "XGBoost")
print(f"🎯 AUC (XGBoost): {roc_auc_xgb:.4f}")
save_pickle(best_xgb, "models/xgboost.pkl")
compress_model("models/xgboost.pkl", "models/xgboost.tar.gz")

import json

# Save metadata for MLflow
metadata = {
    "rf": {
        "params": rf_grid.best_params_,
        "roc_auc": roc_auc_rf,
        "f1_score": f1_score(y_test, y_pred_rf)
    },
    "xgb": {
        "params": xgb_rand.best_params_,
        "roc_auc": roc_auc_xgb,
        "f1_score": f1_score(y_test, y_pred_xgb)
    }
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("✅ Saved metadata for MLflow at models/metadata.json")

