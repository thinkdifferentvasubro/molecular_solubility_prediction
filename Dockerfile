FROM python:3.12.9

WORKDIR /app

COPY requirements.txt .
COPY src/serving/ /app/src/serving
COPY api/ /app/api

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["uvicorn", "app/application:app", "--host", "0.0.0.0", "--port", "8080"]
