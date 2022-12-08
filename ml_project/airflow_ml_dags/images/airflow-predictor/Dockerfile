FROM python:3.9.7-slim
COPY predict.py requirements.txt /
COPY ../../src/train.py src/
COPY ../../src/predict.py src/
COPY ../../src/net.py src/
COPY ../../model/* model/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "predict.py"]
