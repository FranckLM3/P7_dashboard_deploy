FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . /app

# Port used by Cloud Run
ENV PORT 8080
EXPOSE ${PORT}

# APP_MODULE should be something like `main:app` or `app:app` depending on your code
ENV APP_MODULE main:app

CMD exec uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}
