import streamlit as st
import pandas as pd
import pickle
import mlflow
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

# === Set MLflow Tracking URI ===
mlflow.set_tracking_uri("file:///workspaces/byr_uc/mlruns")

# === Load Encoded Columns (used during training) ===
with open("models/encoded_columns.pkl", "rb") as f:
    encoded_columns = pickle.load(f)

# === Load Model ===
@st.cache_resource
def load_model(model_type="rf"):
    path = "models/random_forest.pkl" if model_type == "rf" else "models/xgboost.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

# === Streamlit Page Setup ===
st.set_page_config("Clinical Trial Predictor", layout="centered")

# === Header ===
colored_header(
    label="🎯 Clinical Trial Success Predictor",
    description="Predict the success of clinical trials based on key features.",
    color_name="violet-70"
)

# === Model Selection ===
model_choice = st.radio("🤖 Select Model", ["Random Forest", "XGBoost"], horizontal=True)
model_type = "rf" if model_choice == "Random Forest" else "xgb"
model = load_model(model_type)

# === Input Form ===
with st.form("input_form"):
    st.subheader("📥 Input Trial Details")

    col1, col2 = st.columns(2)

    with col1:
        enrollment = st.number_input("Enrollment (in 1000s)", min_value=0)
        duration = st.number_input("Duration (in days)", min_value=1)
        phase = st.selectbox("Trial Phase", options=[2, 3, 4])
        sponsor = st.selectbox("Sponsor Type", options=["NIH", "Industry", "University", "Other"])

    with col2:
        gender = st.selectbox("Gender", options=["male", "female", "all"])
        condition = st.selectbox("Condition", options=["cancer", "diabetes", "cardio", "covid"])
        location = st.selectbox("Location", options=["us", "canada"])

    submitted = st.form_submit_button("🚀 Predict")

# === Handle Submission ===
if submitted:
    # Create raw input DataFrame
    raw_input = pd.DataFrame([{
        "enrollment": enrollment,
        "duration": duration,
        "phase": phase,
        "sponser_type": sponsor,
        "gender": gender,
        "condition": condition,
        "location": location
    }])

    # Encode input using one-hot and align with training columns
    input_encoded = pd.get_dummies(raw_input, drop_first=True, dtype=int)
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    pred_text = "✅ Success" if prediction == 1 else "❌ Failure"

    # Display Prediction
    st.metric(label="Prediction", value=pred_text)
    style_metric_cards(background_color="#f0f2f6", border_size_px=1)

    # Show Input Summary
    st.markdown("---")
    st.subheader("🔍 Input Summary")

    def highlight_missing(val):
        return 'background-color: red' if pd.isnull(val) else ''

    st.dataframe(raw_input.style.applymap(highlight_missing))
