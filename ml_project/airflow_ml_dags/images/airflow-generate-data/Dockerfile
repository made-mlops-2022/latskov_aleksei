FROM python:3.9.7-slim
COPY data_generation.py requirements.txt test.csv /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "data_generation.py"]
