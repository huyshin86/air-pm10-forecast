# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by osmium
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the runtime components only
COPY run_model.py .
COPY src/ ./src/

# Make sure Python can find the src module
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command: run the forecast script
ENTRYPOINT ["python", "run_model.py"]