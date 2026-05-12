FROM python:3.12.9-slim

WORKDIR /app

COPY requirements.txt .

COPY src/serving/ /app/src/serving
COPY api/ /app/api
COPY src/Data/load.py /app/src/Data/load.py

RUN pip install --no-cache-dir -r requirements-docker.txt

EXPOSE 8080

CMD ["uvicorn", "api.application:app", "--host", "0.0.0.0", "--port", "8080"]