"""Container and CI wiring.

These guard the invariants the deployment shape relies on, without needing a Docker
daemon: the image carries the committed models but never retrains, the entrypoint only
generates a world and then serves on all interfaces, the compose file points the gateway
at the Redis service and persists the world, CI runs generate -> train -> test in that
order, and shell scripts stay LF so they still execute after a Windows checkout.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
ENTRYPOINT = (ROOT / "scripts" / "docker-entrypoint.sh").read_bytes()
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")


def _copied(dockerfile: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^COPY\s+(\S+)\s+\S+", dockerfile, flags=re.M)}


def test_image_ships_code_and_committed_artifacts_only():
    src = _copied(DOCKERFILE)
    assert {"requirements.txt", "src", "scripts", "config", "artifacts/models", "artifacts/reports"} <= src
    assert not any(s.startswith("artifacts/data") or s in (".", "artifacts") for s in src), src
    assert DOCKERFILE.startswith("# ") and "FROM python:3.12-slim" in DOCKERFILE
    assert "libgomp1" in DOCKERFILE, "LightGBM needs OpenMP at runtime"
    assert "USER chakra" in DOCKERFILE, "the gateway must not run as root"


def test_container_never_retrains():
    for text, name in ((DOCKERFILE, "Dockerfile"), (ENTRYPOINT.decode(), "entrypoint"), (COMPOSE, "compose")):
        assert "02_train" not in text, f"{name} would retrain and desynchronise the committed model from the code"


def test_entrypoint_generates_then_serves():
    text = ENTRYPOINT.decode("utf-8")
    assert ENTRYPOINT.startswith(b"#!/bin/sh\n")
    assert b"\r" not in ENTRYPOINT, "CRLF would break the shebang inside the container"
    assert "orders.pkl" in text and "01_generate_data.py" in text
    assert re.search(r"exec python scripts/serve\.py --host 0\.0\.0\.0", text)
    assert text.index("01_generate_data.py") < text.index("serve.py")
    assert 'ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]' in DOCKERFILE


def test_gitattributes_pins_shell_scripts_to_lf():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^\*\.sh\s+text\s+eol=lf", attrs, flags=re.M)


def test_compose_wires_redis_and_persists_the_world():
    assert re.search(r"^\s+api:\s*$", COMPOSE, flags=re.M) and re.search(r"^\s+redis:\s*$", COMPOSE, flags=re.M)
    assert "image: redis:7-alpine" in COMPOSE
    assert "REDIS_URL: redis://redis:6379/0" in COMPOSE
    assert "CHAKRA_EPSILON:" in COMPOSE
    assert '"8080:8080"' in COMPOSE
    assert re.search(r"^\s+- chakra-data:/app/artifacts/data\s*$", COMPOSE, flags=re.M)
    assert re.search(r"^volumes:\s*\n\s+chakra-data:", COMPOSE, flags=re.M)


def test_build_context_excludes_world_and_repo_metadata():
    lines = {ln.strip() for ln in DOCKERIGNORE.splitlines() if ln.strip() and not ln.startswith("#")}
    assert {".git", "artifacts/data/", "tests/", "**/__pycache__/"} <= lines


def test_ci_runs_generate_train_test_in_order():
    assert 'python-version: "3.12"' in CI and "cache: pip" in CI
    assert re.search(r"^on:\s*\n\s+push:\s*\n\s+pull_request:", CI, flags=re.M)
    steps = ["pip install -r requirements.txt",
             "scripts/01_generate_data.py",
             "02_train.py",
             "python -m pytest tests -q"]
    positions = [CI.index(s) for s in steps]
    assert positions == sorted(positions), "CI must install, generate, train, then test"
    assert "CHAKRA_ARTIFACTS:" in CI, "CI must train into a scratch artifacts dir, not over the committed models"
    assert "docker build" in CI and "/healthz" in CI


def test_serving_requirements_have_no_torch():
    names = {re.split(r"[<>=\[ ]", ln.strip(), maxsplit=1)[0].lower() for ln in REQUIREMENTS.splitlines() if ln.strip()}
    assert "torch" not in names and "torch-geometric" not in names
    for needed in ("lightgbm", "onnxruntime", "fastapi", "uvicorn", "redis", "networkx", "pytest"):
        assert needed in names
