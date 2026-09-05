"""Run the risk gateway.  python scripts/serve.py [--port 8080] [--reload]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    uvicorn.run("chakrashield.serving.app:app", host=a.host, port=a.port, reload=a.reload, log_level="info",
                access_log=False, workers=1)


if __name__ == "__main__":
    main()
