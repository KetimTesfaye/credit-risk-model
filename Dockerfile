# Use a lightweight, stable python base image
FROM python:3.10-slim

# Establish production environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install operating system utility baseline requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list file over first to optimize Docker cache layers
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Bring over the remaining code assets across local architecture folders
COPY . /app/

# Expose FastAPI's default runtime communication port
EXPOSE 8000

# Start FastAPI via the Uvicorn production ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]