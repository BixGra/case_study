import json

from fastapi import FastAPI
from starlette.requests import Request

from src.tools.models.full_model import FullModel

app = FastAPI()
full_model = FullModel()


@app.get("/")
async def get_method():
    return "ok"


@app.post("/predict")
async def post_method(request: Request):
    data = await request.body()
    data_ = json.loads(data)
    print(data_)
    duration = data_["duration"]
    output = full_model.predict(duration)
    return output
