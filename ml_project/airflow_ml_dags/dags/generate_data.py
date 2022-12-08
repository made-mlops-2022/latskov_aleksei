from datetime import datetime, timedelta
from airflow import DAG

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


default_args = {
    'owner': 'Latskov Alexcei',
    'email': ['alexcei64rus@mail.ru'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        'generate_data',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2022, 12, 4)
) as dag:
    generate_data = DockerOperator(
        image='airflow-generate-data',
        command='--output-dir /data/raw/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-generate-data',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source='/', target='/data', type='bind')]
    )
