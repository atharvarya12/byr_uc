from fastapi import APIRouter
import pandas as pd

router = APIRouter(prefix="/eda", tags=["EDA"])

@router.get("/summary")
def eda_summary():
    df = pd.read_csv("ProcessedData1.csv")
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(5).to_dict(orient="records")
    }
