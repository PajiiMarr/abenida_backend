FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    R_HOME=/usr/lib/R \
    LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc g++ gfortran \
    libgdal-dev libgeos-dev libproj-dev proj-data proj-bin gdal-bin \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('earth', repos='https://cloud.r-project.org', quiet=TRUE)" \
    && R -e "library(earth); cat('earth', as.character(packageVersion('earth')), '\n')"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY app ./app

COPY register_sar_to_dem.py .

COPY datas ./datas

COPY outputs/phase3 ./outputs/phase3

COPY outputs/phase2 ./outputs/phase2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]