#!/bin/sh
# Container entrypoint: make sure a replay world exists, then serve.
#
# Models and reports ship in the image; the replay pickles do not (gitignored,
# 77 MB for the default 60k world), so the first start on an empty data volume
# generates a smaller one. Generation only -- training is never run here, the
# committed model is the one whose feature list matches the code.
#
# Env knobs: CHAKRA_WORLD_ORDERS / CHAKRA_WORLD_CUSTOMERS / CHAKRA_WORLD_RINGS
# size the generated world; PORT sets the listen port (default 8080).
# Any arguments replace the server command (e.g. a one-off script run).
set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

ARTIFACTS="${CHAKRA_ARTIFACTS:-/app/artifacts}"
if [ ! -f "$ARTIFACTS/data/orders.pkl" ]; then
    echo "[entrypoint] no replay world under $ARTIFACTS/data; generating one"
    python scripts/01_generate_data.py \
        --orders "${CHAKRA_WORLD_ORDERS:-12000}" \
        --customers "${CHAKRA_WORLD_CUSTOMERS:-4000}" \
        --rings "${CHAKRA_WORLD_RINGS:-12}"
else
    echo "[entrypoint] using existing replay world under $ARTIFACTS/data"
fi

exec python scripts/serve.py --host 0.0.0.0 --port "${PORT:-8080}"
