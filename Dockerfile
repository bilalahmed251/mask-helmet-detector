# Use a standard, light Python 3.11 base image
FROM python:3.11-slim

# Set system variables for Python and security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker build cache
COPY requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Create a non-root user for production security guidelines
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports for both services
# 8000 for FastAPI API, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# Default command: run the API (can be overridden in docker-compose.yml)
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
