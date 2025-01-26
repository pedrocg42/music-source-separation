# Use the official Python image from Docker Hub
FROM python:3.11

# Create a working directory
WORKDIR /app

# Update the package list and install system-level dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Clean up after apt-get to reduce the image size
RUN apt-get clean

# Copy your application code to the working directory
COPY requirements.txt .
COPY pyproject.toml .
COPY src/ .

# Install dependencies
RUN pip install uv
RUN uv pip install --no-cache-dir --upgrade pip setuptools
RUN uv pip install --no-cache-dir .
