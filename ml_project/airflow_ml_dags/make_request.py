import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("data/test.csv")
    del data['4']
    for i in range(10):
        request_data = data.iloc[i].tolist()
        response = requests.get(
            "http://localhost:8000/predict/",
            json={"data": [request_data]},
        )
        print(response.status_code, flush=True)
        print(response.json(), flush=True)