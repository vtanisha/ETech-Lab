FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required for ML & PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (ignoring versions that might conflict during production containerization)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scipy joblib threadpoolctl sympy filelock fsspec networkx jinja2 psycopg2-binary torch rtdl scikit-learn shap --no-deps

COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn safely binding to all interfaces inside the Docker network
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
