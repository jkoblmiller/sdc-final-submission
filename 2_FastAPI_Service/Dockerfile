FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first so Docker can cache the installed dependencies
COPY requirements.txt ./requirements.txt

# Install system deps and PyTorch CPU wheels, then install the rest of requirements
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
    pip install --upgrade pip setuptools wheel && \
    # Ensure a NumPy 1.x wheel is installed so compiled extensions built against NumPy 1.x work
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy remaining application files
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
