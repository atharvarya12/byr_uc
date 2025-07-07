from fastapi import APIRouter
from src.train_model import train_all_models  # <- make sure this function exists

router = APIRouter(prefix="/train", tags=["Training"])

@router.post("/")
def train_models():
    result = train_all_models()
    return {"message": "Training completed", "results": result}
