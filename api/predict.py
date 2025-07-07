from fastapi import APIRouter
from api.schema import PredictionInput
import joblib
import numpy as np

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/randomforest")
def predict_rf(data: PredictionInput):
    model = joblib.load("models/random_forest.pkl")
    features = np.array([[data.enrollment, data.duration, data.phase,
                          data.sponser_type, data.gender, data.condition, data.location]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}

@router.post("/xgboost")
def predict_xgb(data: PredictionInput):
    model = joblib.load("models/xgboost.pkl")
    features = np.array([[data.enrollment, data.duration, data.phase,
                          data.sponser_type, data.gender, data.condition, data.location]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
