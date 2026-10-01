FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home sim && mkdir -p /app/logs /app/data/missions && chown -R sim /app/logs /app/data
USER sim

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# Bind to all interfaces inside the container; everything else comes from configs/simulation.yaml.
CMD ["python", "-m", "simulation", "--host", "0.0.0.0", "--port", "8000"]
