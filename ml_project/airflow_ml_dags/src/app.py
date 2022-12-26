import os
import uvicorn
from typing import List, Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel, conlist

from predict import *


app = FastAPI()


class OutputDataModel(BaseModel):
    result: int

class InputDataModel(BaseModel):
    data: List[conlist(Union[float, str, None, int], min_items=29, max_items=29)]


def load_object(path: str) -> Model:
    with open(path + 'model.pickle', "rb") as f:
        return pickle.load(f)


@app.get('/')
def home():
    return {"msg": "OK"}


@app.get('/health')
def rood():
    return {"health check": "200"}


@app.on_event("startup")
def load_model():
    global model
    model_path = PATH_MODEL
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        raise RuntimeError(err)

    model = load_object(model_path)
    print('startup done')


@app.get("/predict_from_file/")
async def predict_from_file():
    predict()
    return {"msg": "done"}


@app.get("/predict/", response_model=OutputDataModel)
async def predict_once(request: InputDataModel):
    data = pd.DataFrame(request.data)
    data.to_csv('test_.csv')
    predict = model.predict(data)
    return {'result': int(predict[0])}


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
