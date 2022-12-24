from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from utils import preparation_data, train_data, predict_data, evaluate_data


with DAG("dag_train",
    start_date=datetime(2022, 11, 14),
    schedule_interval='@weekly', catchup=False) as dag:

    data_preparation = PythonOperator(
        task_id='preparation_id',
        python_callable=preparation_data
    )


    data_train = PythonOperator(
        task_id='train_id',
        python_callable=train_data
    )


    data_predict = PythonOperator(
        task_id='predict_id',
        python_callable=predict_data
    )

    data_evaluate = PythonOperator(
        task_id='evaluate_id',
        python_callable=evaluate_data
    )

    data_preparation >> data_train >> data_predict >> data_evaluate