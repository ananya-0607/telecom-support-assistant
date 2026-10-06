# Build separate CPU backend and lightweight Streamlit images.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

FROM base AS api
COPY requirements.txt .
# CPU wheels avoid downloading CUDA libraries for the local embedding model.
RUN python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY dataset_creation ./dataset_creation
ENV PYTHONPATH=/app/src HF_HOME=/app/data/indexes/huggingface_cache
RUN mkdir -p data config logs && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "telecom_support.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM base AS ui
RUN python -m pip install streamlit==1.45.1 "httpx>=0.27,<1"
COPY app.py .
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]
