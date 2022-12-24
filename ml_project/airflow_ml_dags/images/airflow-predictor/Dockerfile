FROM python:3.9.7-slim

COPY requirements.txt /
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY data_/ /data_/
COPY model/ /model/
COPY predict.py config.py net.py train.py /

CMD ["python3", "predict.py"]
