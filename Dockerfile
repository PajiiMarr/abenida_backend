# backend/Dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    R_HOME=/usr/lib/R \
    LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu

# ---- System deps ------------------------------------------------------------
# git: required for pip to install editable VCS dependencies (pyGWRx)
# build-essential + gfortran: wheels that need compiling (mgwr deps, rpy2)
# libgdal/libgeos/libproj: rasterio, geopandas, pyproj, spglm
# r-base + r-base-dev: rpy2 + earth backend for MARS (used by m3/m4/m5)
# curl/ssl/xml2: R package installs and general HTTP
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc g++ gfortran \
    libgdal-dev libgeos-dev libproj-dev proj-data proj-bin gdal-bin \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---- R 'earth' package (MARS backend for m3, m4, m5) -----------------------
RUN R -e "install.packages('earth', repos='https://cloud.r-project.org', quiet=TRUE)" \
    && R -e "library(earth); cat('earth', as.character(packageVersion('earth')), '\n')"

# ---- Python deps ------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ---- App + artifacts --------------------------------------------------------
# App tree — includes app/training/models.py where m1–m5 live.
COPY app ./app

# Root-level scripts for the background retraining pipeline.
# train_model.py was removed: it imported a non-existent app.training.trainer
# and is superseded by run_suite.py, which drives m1–m5 via train_model_suite().
COPY register_sar_to_dem.py .
COPY run_suite.py .
# COPY retrain_new_event.py .
# COPY create_event_datasets.py .

# Phase 3 pickles + metrics. After retraining on the host, this tree contains:
#   m1_gwpca_svm, m2_gwr_mgwr, m3_gwpca_mars, m4_stacking_all,
#   m5_gwr_mars_svm_pipeline          (one .pkl + _summary.json per category)
COPY outputs/phase3 ./outputs/phase3

# Phase 2 rasters, preprocessed arrays, feature CSVs, and feature_names.json
# needed by run_suite.py. .dockerignore trims the large intermediates.
COPY outputs/phase2 ./outputs/phase2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]