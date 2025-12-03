# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy project files
COPY . .

# Create directories for input/output
RUN mkdir -p input output temp_output

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_API_KEY=""

# Default command
CMD ["python", "run.py", "--image_paths", "input/image.jpg", "--output_path", "output"]
