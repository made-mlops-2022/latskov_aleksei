import os
import uvicorn

from fastapi import FastAPI
from predict import *


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/")
async def read_item():
    predict()


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
