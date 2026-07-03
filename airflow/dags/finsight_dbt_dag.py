from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# dbt project is mounted into the container at /opt/dbt
DBT_DIR = "/opt/dbt"
DBT_PROFILES = "/opt/dbt"

default_args = {
    "owner": "nika",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="finsight_dbt_pipeline",
    description="Run the FinSight dbt transformations on BigQuery",
    default_args=default_args,
    schedule="0 6 * * *",          # daily 06:00 (UTC in Airflow)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["dbt", "bigquery", "finsight"],
) as dag:

    dbt_debug = BashOperator(
        task_id="dbt_debug",
        bash_command=(
            f"cd {DBT_DIR} && "
            f"dbt debug --profiles-dir {DBT_PROFILES}"
        ),
    )

    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=(
            f"cd {DBT_DIR} && "
            f"dbt build --no-partial-parse --profiles-dir {DBT_PROFILES}"
        ),
    )

    dbt_debug >> dbt_build
