from fastapi import FastAPI
from api import eda, train, predict

app = FastAPI(title="Clinical Trial ML API")

app.include_router(eda.router)
app.include_router(train.router)
app.include_router(predict.router)
