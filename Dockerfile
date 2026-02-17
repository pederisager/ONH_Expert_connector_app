FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.deploy.txt

COPY . .

# Build a local vector index into the image so deploy/runtime never depends on
# checked-in data/index artifacts.
RUN python3 -m app.index.build --index-root data/index

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
