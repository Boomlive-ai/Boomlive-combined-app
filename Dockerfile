# # Use a Python 3.12 slim image
# FROM python:3.12

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     FLASK_APP=app.py \
#     FLASK_RUN_HOST=0.0.0.0 \
#     FLASK_RUN_PORT=5000

# # Set the working directory
# WORKDIR /app

# # Install system dependencies and Tesseract OCR
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         curl \
#         libgl1 \
#         g++ \
#         python3-dev \
#         libssl-dev \
#         libffi-dev \
#         libxml2-dev \
#         libxslt1-dev \
#         zlib1g-dev \
#         libjpeg-dev \
#         libpng-dev \
#         tesseract-ocr \
#         tesseract-ocr-eng \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
#     && pip install --no-cache-dir cython==3.0.0 

# # Copy requirements file
# COPY requirements.txt /app/

# # Install remaining Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . /app

# # Expose port 5000
# EXPOSE 5000

# # Command to run the application
# CMD ["flask", "run"]

# Use a Python 3.12 slim image
FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

# Set the working directory
WORKDIR /app

# Install system dependencies and Tesseract OCR + FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        g++ \
        python3-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \    # ✅ Install FFmpeg (includes ffprobe)
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir cython==3.0.0 

# Copy requirements file
COPY requirements.txt /app/

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["flask", "run"]
