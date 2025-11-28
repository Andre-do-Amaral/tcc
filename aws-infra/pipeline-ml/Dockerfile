FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY utils utils
COPY pipeline-ml.py .

ENTRYPOINT ["python", "-u", "pipeline-ml.py"]