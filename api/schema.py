from pydantic import BaseModel

class PredictionInput(BaseModel):
    enrollment: float
    duration: float
    phase: int
    sponser_type: int
    gender: int
    condition: int
    location: int
