
# Base image with CUDA 12.8 and cuDNN 9 support
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl vim && \
    rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Create working directory
WORKDIR /workspace

# Copy dependencies file and install them
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /workspace/
