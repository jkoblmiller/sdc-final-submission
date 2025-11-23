FROM python:3.11-slim

WORKDIR /app

# Copy service files (build context should be the service folder)
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
