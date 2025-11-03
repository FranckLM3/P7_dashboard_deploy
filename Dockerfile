FROM python:3.9-slim

# Install system dependencies (OpenMP for LightGBM/Sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app source (includes ressource/ with pipeline.joblib)
COPY . /app

# Streamlit configuration
ENV PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE ${PORT}

# Default command: run Streamlit dashboard
CMD streamlit run dashboard.py --server.port=${PORT} --server.address=0.0.0.0
