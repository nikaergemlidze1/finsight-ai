FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DBT_PROFILES_DIR=/app \
    GOOGLE_CLOUD_PROJECT=finsight-data-engineering

WORKDIR /app

RUN pip install --no-cache-dir dbt-bigquery==1.11.1

COPY . /app

CMD ["dbt", "build", "--no-partial-parse"]
