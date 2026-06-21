# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages with CPU-only wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application files
COPY . .

# Ensure data directory exists and pre-populate with metadata CSVs and split files
RUN mkdir -p /app/data && \
    cp BBox_List_2017.csv Data_Entry_2017.csv test_list.txt train_val_list.txt /app/data/ 2>/dev/null || true

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
