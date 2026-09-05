"""Real orders through the whole pipeline: Amazon India seller sale report, Q2 2022.

    python scripts/14_real_data.py                 # downloads the CSV (19 MB) on first run
    python scripts/14_real_data.py --csv path.csv  # or point at a local copy

The dataset (Kaggle "Unlock Profits with E-Commerce Sales Data"; 128,976 order lines from
31 Mar to 29 Jun 2022) is the closest public thing to a merchant's COD export: real Indian
PIN codes, real basket values, real final outcomes including "Rejected by Buyer" and
"Returned to Seller". It is also missing most of what ChakraShield's strongest features
need, and this script does not pretend otherwise:

* no customer, device, address or payment-mode fields -- every order is a new customer with a
  city/state/PIN address, so velocity, graph and address-defect features carry no signal and
  the label is *return-to-seller risk on marketplace orders*, not pure COD refusal;
* no order hour, no coupon flag, no parcel weight (proxied from the product category);
* margin and CAC are the repo defaults, so the rupee results are illustrative economics on
  real outcomes rather than the seller's own P&L.

What it does measure honestly: on ~31k orders with a final outcome, what PIN priors, basket
and channel are worth (AUC / PR-AUC), whether isotonic calibration and class-conditional
conformal sets hold on real labels, and how the three-action resolver behaves against
binary cut-offs when the base rate is 7 % instead of the synthetic world's 26 %.

Everything lands in artifacts_real/ (gitignored); a summary is written to
artifacts/reports/real_amazon_2022.json so the README numbers are reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chakrashield.data.adapter import load_merchant_orders
from chakrashield.data.pipeline import materialise

SOURCE_URL = "https://raw.githubusercontent.com/shivamverma26/Amazon_Sales_Analysis/main/Amazon%20Sale%20Report.csv"
REAL_ARTIFACTS = ROOT / "artifacts_real"
MAPPING = ROOT / "config" / "amazon_sale_report.mapping.json"
SUMMARY = ROOT / "artifacts" / "reports" / "real_amazon_2022.json"

# Parcel weight proxy by product category (grams). The report has no weight column and the
# logistics cost in C_FN is weight-tiered, so a category-level guess is better than one constant.
WEIGHT_BY_CATEGORY = {"T-shirt": 250, "Shirt": 300, "Blazzer": 900, "Trousers": 450, "Perfume": 350, "Wallet": 200,
                      "Socks": 100, "Shoes": 800, "Watch": 300, "Kurta": 350, "Saree": 500, "Set": 600,
                      "Western Dress": 400, "Top": 250, "Ethnic Dress": 500, "Bottom": 400, "Dupatta": 200}


def download(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, csv_path)
    print(f"[download] {csv_path.stat().st_size / 1e6:.1f} MB")


def derive(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add the columns the adapter needs that the report does not have, and say what was derived."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    # each order line is its own customer: a synthetic 10-digit key from the order id, never a real phone
    df["Derived Phone"] = df["Order ID"].astype(str).map(lambda s: "9" + hashlib.sha1(s.encode()).hexdigest()[:9].translate(str.maketrans("abcdef", "123456")))
    pin = pd.to_numeric(df["ship-postal-code"], errors="coerce")
    df["ship-postal-code"] = pin.map(lambda v: "" if pd.isna(v) else f"{int(v):06d}")
    df["Derived Address"] = (df["ship-city"].fillna("").astype(str).str.title() + ", " + df["ship-state"].fillna("").astype(str).str.title()
                             + " " + df["ship-postal-code"]).str.strip(", ")
    df["Derived Weight (g)"] = df["Category"].map(WEIGHT_BY_CATEGORY).fillna(400.0)
    df["Derived Payment Mode"] = "COD"        # the report has no payment column; see the module docstring
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    before = len(df)
    df = df[df["Amount"].notna() & (df["Amount"] > 0) & (df["Qty"] >= 1) & (df["ship-postal-code"] != "")]
    notes = {
        "rows_in": int(before), "rows_with_amount_qty_pin": int(len(df)),
        "derived": {"customer_phone": "sha1(order id): one customer per order line", "shipping_address": "city, state PIN",
                    "weight_grams": "category proxy", "payment_method": "constant COD (column absent)"},
        "absent": ["customer identity", "device / IP", "street address", "payment mode", "order hour", "coupon", "checkout duration",
                   "merchant margin", "CAC"],
        "unused_columns": ["Fulfilment", "ship-service-level", "Category", "Size", "Courier Status", "B2B"],
        "status_counts": raw["Status"].value_counts().to_dict(),
    }
    return df, notes


def run(cmd: list[str], env: dict) -> str:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    if p.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"{' '.join(cmd[1:])} failed with {p.returncode}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REAL_ARTIFACTS / "source" / "amazon_sale_report.csv"))
    ap.add_argument("--skip-eval", action="store_true", help="stop after training")
    a = ap.parse_args()
    csv_path = Path(a.csv)
    if not csv_path.exists():
        download(csv_path)
    t0 = time.time()
    raw = pd.read_csv(csv_path, low_memory=False)
    derived, notes = derive(raw)
    derived_csv = REAL_ARTIFACTS / "source" / "amazon_sale_report.derived.csv"
    derived.to_csv(derived_csv, index=False)

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    df, rep = load_merchant_orders(derived_csv, mapping)
    print(f"[ingest] {rep['rows_in']:,} rows -> {rep['rows_out']:,} labelled orders | return rate {rep['cod_rto_rate']:.1%} | "
          f"dropped {rep['dropped']} | defaults used {rep['defaults_used']}")
    data_dir = REAL_ARTIFACTS / "data"
    world = materialise(df, data_dir, extra_world={"source": "amazon_india_q2_2022", "ingest_report": rep, "derivation": notes})
    print(f"[replay] {world['orders']:,} orders | graph {world['graph']} | {time.time() - t0:.0f}s")

    env = {**os.environ, "CHAKRA_ARTIFACTS": str(REAL_ARTIFACTS), "PYTHONUTF8": "1"}
    t1 = time.time()
    out = run([sys.executable, str(ROOT / "scripts" / "02_train.py")], env)
    print("\n".join(ln for ln in out.splitlines() if ln.startswith(("[data]", "[cand", "[select]", "[onnx]"))))
    print(f"[train] {time.time() - t1:.0f}s")
    summary = json.loads((REAL_ARTIFACTS / "models" / "training_summary.json").read_text(encoding="utf-8"))
    result = {
        "source": {"name": "Amazon India seller sale report, Q2 2022", "url": SOURCE_URL, "rows": notes["rows_in"]},
        "derivation": {k: v for k, v in notes.items() if k != "status_counts"}, "status_counts": notes["status_counts"],
        "ingest": rep, "world": {k: v for k, v in world.items() if k != "ingest_report"},
        "model": {k: summary["chakra"][k] for k in ("gamma", "auc", "pr_auc", "brier", "ece_raw", "ece_calibrated", "best_iter", "conformal")},
        "candidates": summary["candidates"], "onnx_parity": summary["onnx"]["parity_max_abs_diff"],
        "feature_importance_top": dict(list(summary["chakra"]["feature_importance"].items())[:8]),
    }
    if not a.skip_eval:
        t2 = time.time()
        out = run([sys.executable, str(ROOT / "scripts" / "03_evaluate.py")], env)
        print("\n".join(ln for ln in out.splitlines() if ln.startswith(("policy", "ALLOW_ALL", "BASE@", "CHAKRA", "ORACLE", "[budget]", "[sensitivity]"))))
        print(f"[evaluate] {time.time() - t2:.0f}s")
        ev = json.loads((REAL_ARTIFACTS / "reports" / "evaluation.json").read_text(encoding="utf-8"))
        result["evaluation"] = {
            "test_orders": ev["test_orders"], "test_gmv": ev["test_gmv"], "test_rto_rate": ev.get("test_rto_rate"),
            "policies": {k: {kk: v[kk] for kk in ("pnl_total", "delta_vs_allow_all", "actions", "good_customers_lost_expected", "rto_shipped_expected")}
                         for k, v in ev["policies"].items()},
            "conformal_test": ev["conformal_test"], "certainty": ev["certainty"], "thresholds": ev["thresholds"],
            "friction_budget": ev.get("friction_budget"), "sensitivity_wins": ev.get("sensitivity_wins"),
        }
    SUMMARY.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"[done] summary -> {SUMMARY} | artifacts in {REAL_ARTIFACTS} | {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
