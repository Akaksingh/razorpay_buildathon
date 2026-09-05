"""Ingest a merchant's order export and materialise the same artifacts the synthetic world produces.

    python scripts/ingest_csv.py --csv orders_export.csv --mapping config/merchant_schema.example.json
    python scripts/02_train.py && python scripts/03_evaluate.py && python scripts/serve.py

Write a starter mapping with --write-mapping <path>, edit the column names and status values, and
re-run. The ingestion report lists every default that had to be used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chakrashield.config import DATA_DIR
from chakrashield.data.adapter import load_merchant_orders, write_mapping_example
from chakrashield.data.pipeline import materialise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--mapping")
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--write-mapping", help="write an example mapping file here and exit")
    a = ap.parse_args()
    if a.write_mapping:
        write_mapping_example(a.write_mapping)
        print(f"[mapping] example written to {a.write_mapping}")
        return
    if not (a.csv and a.mapping):
        ap.error("--csv and --mapping are required (or --write-mapping)")
    mapping = json.loads(Path(a.mapping).read_text(encoding="utf-8"))
    df, rep = load_merchant_orders(a.csv, mapping)
    print(f"[ingest] {rep['rows_in']:,} rows -> {rep['rows_out']:,} labelled orders | COD share {rep['cod_share']:.1%} "
          f"COD RTO {rep['cod_rto_rate']:.1%} | dropped {rep['dropped']} | defaults used {rep['defaults_used']}")
    if len(df) < 500:
        print("[warn] fewer than 500 labelled orders: the chronological splits in 02_train.py will be thin")
    world = materialise(df, Path(a.out), extra_world={"source": "merchant_csv", "ingest_report": rep})
    print(f"[done] artifacts in {a.out}: {world['orders']:,} orders, graph {world['graph']}")


if __name__ == "__main__":
    main()
