#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import tarfile

#load preprocessed data
df = pd.read_csv('/workspaces/byr_uc/ProcessedData1.csv')
df = df.drop(df.columns[0], axis = 1)

# Assuming df is your DataFrame
X = df.drop(columns=[' final_status_success'])
y = df[' final_status_success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy:  {acc:.4f}")

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save as .pkl
joblib.dump(model, 'random_forest_model.pkl')

# Create a .tar archive containing the .pkl file
with tarfile.open('random_forest_model.tar', 'w') as tar:
    tar.add('random_forest_model.pkl')