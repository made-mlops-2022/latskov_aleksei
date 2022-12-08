from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from datetime import timedelta
from docker.types import Mount
import os

default_args = {
    'owner': 'Latskov Alexcei',
    'email': ['alexcei64rus@mail.ru'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    "predictor",
    default_args=default_args,
    schedule_interval="@daily"
):
    wait_data = PythonSensor(
        task_id="wait_data",
        python_callable=os.path.exists,
        op_args=["/opt/airflow/data/ready/{{ ds }}/data.csv"],
        timeout=5000,
        mode="poke",
        poke_interval=60,
        retries=100,
    )

    predictor = DockerOperator(
        image="airflow-predictor",
        command="--data-dir  /data/val/{{ ds }} --model-dir /data/models/{{ ds }} --result-dir /data/predictions/{{ ds }}]",
        network_mode="bridge",
        task_id="docker-airflow-predictor",
        do_xcom_push=False,
        auto_remove=True,
        mounts = [Mount(source='/', target='/data', type='bind')]
    )

    wait_data >> predictor