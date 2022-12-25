FROM python:3.9.7-slim
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY data/ /data/
COPY logs/ logs/
COPY model/ model/
COPY src/ src/
COPY make_request.py .

WORKDIR .

CMD ["python", "src/app.py"]
