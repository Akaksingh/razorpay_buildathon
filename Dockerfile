# ChakraShield risk gateway.
#
# The image ships the code plus the *committed* artifacts (trained models and
# evaluation reports). It never retrains: the served model's feature list must
# match the code it was trained against, and that pairing is what the repo
# commits. The replay world (artifacts/data/*.pkl) is gitignored and regenerable,
# so the entrypoint builds a small one on first start when the data volume is
# empty, then serves. torch is not needed for serving and is not installed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHAKRA_ARTIFACTS=/app/artifacts

# libgomp: LightGBM's OpenMP runtime (used for TreeSHAP reason codes on the request path)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY artifacts/models ./artifacts/models
COPY artifacts/reports ./artifacts/reports

# Unprivileged runtime user; artifacts/data is the mount point for the world volume
# and must be writable (decision ledger, learned-behaviour snapshot, replay pickles).
RUN groupadd --system chakra && useradd --system --gid chakra --home-dir /app chakra \
    && mkdir -p artifacts/data \
    && chmod +x scripts/docker-entrypoint.sh \
    && chown -R chakra:chakra /app/artifacts
USER chakra

EXPOSE 8080

# start-period covers world generation on an empty volume (about a minute for the default size).
HEALTHCHECK --interval=30s --timeout=3s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz > /dev/null || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
