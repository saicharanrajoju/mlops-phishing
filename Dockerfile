FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# libgomp1 is required by xgboost wheels at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Serving entrypoint. (The MLflow service in docker-compose overrides this command.)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
