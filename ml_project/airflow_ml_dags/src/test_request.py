import pandas as pd
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "OK"}

def test_predict():
    data = pd.read_csv("../data/test.csv").iloc[0]
    del data['4']
    request_data = data.tolist()
    response = client.get("/predict/", json={"data": [request_data]})
    assert response.status_code == 200
    assert response.json() == {'result': 0}

if __name__=='__main__':
    test_home()
