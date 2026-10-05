# backend/Dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    R_HOME=/usr/lib/R \
    LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu

# ---- System deps ------------------------------------------------------------
# build-essential + gfortran: wheels that need compiling (mgwr deps, rpy2)
# libgdal/libgeos/libproj: rasterio, geopandas, pyproj, spglm
# r-base + r-base-dev: rpy2 + earth backend for MARS
# curl/ssl/xml2: R package installs and general HTTP
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc g++ gfortran \
        libgdal-dev libgeos-dev libproj-dev proj-data proj-bin gdal-bin \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---- R 'earth' package (MARS backend) --------------------------------------
RUN R -e "install.packages('earth', repos='https://cloud.r-project.org', quiet=TRUE)" \
    && R -e "library(earth); cat('earth', as.character(packageVersion('earth')), '\n')"

# ---- Python deps ------------------------------------------------------------
WORKDIR /app

# Install Python requirements first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# ---- App + artifacts --------------------------------------------------------
# Copy the whole app tree; .dockerignore excludes training scripts and
# heavy Phase 2 intermediates.
COPY app ./app

# Phase 3 pickles + metrics.
COPY outputs/phase3 ./outputs/phase3

# Phase 2 rasters, preprocessed arrays, feature CSVs, and any optional
# metadata. .dockerignore trims the big intermediates.
COPY outputs/phase2 ./outputs/phase2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]