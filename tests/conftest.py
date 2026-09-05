import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os
import tempfile

# keep test decisions out of the real propensity ledger
os.environ.setdefault("CHAKRA_LEDGER", os.path.join(tempfile.gettempdir(), "chakrashield_test_decisions.jsonl"))
