FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy service files
COPY ../2_FastAPI_Service/requirements.txt ./requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
    pip install --upgrade pip setuptools wheel && \
    # Ensure a NumPy 1.x wheel is installed so compiled extensions built against NumPy 1.x work
    pip install --no-cache-dir "numpy<2" && \
    # Install compatible CPU wheels for PyTorch and torchvision
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy application code after dependencies are installed
COPY ../2_FastAPI_Service/ /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
