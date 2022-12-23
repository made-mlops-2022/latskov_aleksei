import os
import uvicorn
import time

from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

from predict import *


time_start = time.time()
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
    time_finish = time.time()
    time_diff = time_finish - time_start
    if time_diff > 30:
        return {"msg": f"Ok {time_diff}"}
    else:
        raise HTTPException(status_code=503)


@app.get('/health')
def rood():
    time_finish = time.time()
    time_diff = time_finish - time_start
    if time_diff < 120:
        return {"health check": f"Ok {time_diff}"}
    else:
        raise HTTPException(status_code=503)


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
