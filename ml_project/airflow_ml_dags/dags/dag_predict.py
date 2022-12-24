from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta, datetime
from docker.types import Mount


default_args = {
    'owner': 'Latskov Alexcei',
    'email': ['alexcei64rus@mail.ru'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('dag_predict',
         default_args=default_args,
         schedule_interval="@daily",
         start_date=datetime(2022, 12, 4),
) as dag:
    predict = DockerOperator(
        image="airflow-predictor",
        command="",
        network_mode="bridge",
        task_id="docker-airflow-predictor",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts = [Mount(source='/', target='/data', type='bind')]
    )

    predict
