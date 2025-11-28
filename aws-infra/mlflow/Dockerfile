FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
  gcc \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary cryptography

ADD . /app
WORKDIR /app