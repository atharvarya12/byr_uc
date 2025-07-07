import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load your encoded dataset
df = pd.read_csv("ProcessedData.csv")

# Drop Unnamed index column if present
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Step 1: Manually reverse one-hot columns
def reverse_one_hot(df, prefix_list):
    recovered = pd.DataFrame()
    for prefix in prefix_list:
        cols = [col for col in df.columns if col.startswith(prefix + '_')]
        recovered[prefix] = df[cols].idxmax(axis=1).str.replace(prefix + '_', '')
    return recovered

# Step 2: Recover categorical features
prefixes = ['condition', 'gender', 'sponsor_type', 'location']
cat_df = reverse_one_hot(df, prefixes)

# Step 3: Add numeric columns
cat_df['phase'] = df['phase_encoded']
cat_df['enrollment'] = df['enrollment']
cat_df['duration'] = df['duration']

# Step 4: Add label/target column back
cat_df['final_status_success'] = df['final_status_success']  # Adjust this if your label column has a different name

# Step 5: Separate features and labels
X = cat_df.drop(columns=["final_status_success"])
y = cat_df["final_status_success"]

# Step 6: Define categorical columns for encoding
categorical_features = ["phase", "sponsor_type", "gender", "condition", "location"]

# Step 7: Build and fit new preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

X_encoded = preprocessor.fit_transform(X)

# Step 8: Train a new model using same structure
model = RandomForestClassifier()
model.fit(X_encoded, y)

# Step 9: Save model and preprocessor
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
