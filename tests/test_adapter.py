import json

import pandas as pd

from chakrashield.data.adapter import MAPPING_EXAMPLE, load_merchant_orders, write_mapping_example
from chakrashield.data.generator import generate


def export_like_a_merchant(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to merchant-style columns, dd/mm/yyyy dates, status strings, a mix of payment spellings."""
    status = df.rto.map({True: "Returned to Sender", False: "Delivered"})
    pay = df.payment_method.map({"COD": "Cash on Delivery", "UPI": "UPI", "CARD": "Credit Card", "NETBANKING": "Net Banking", "WALLET": "Wallet", "EMI": "EMI"})
    return pd.DataFrame({
        "Order ID": df.order_id, "Order Date": pd.to_datetime(df.ts, unit="s").dt.strftime("%d/%m/%Y %H:%M"),
        "Delivery Date": pd.to_datetime(df.outcome_ts, unit="s").dt.strftime("%d/%m/%Y"),
        "Customer Mobile": "+91 " + df.customer_phone.str[:5] + " " + df.customer_phone.str[5:],
        "Device ID": df.device_fingerprint, "Ship Address": df.shipping_address, "Ship Pincode": df.delivery_pin,
        "Order Amount": df.cart_gmv, "Qty": df.items_count, "Payment Mode": pay, "UTM Source": df.acquisition_channel.str.lower(),
        "Coupon Code": df.coupon_applied.map({True: "SAVE10", False: ""}), "Status": status,
    })


def test_merchant_csv_round_trip(tmp_path):
    src = generate(n_orders=600, seed=11, n_customers=250, n_rings=3, n_shared_addresses=2)
    csv = tmp_path / "export.csv"
    export_like_a_merchant(src).to_csv(csv, index=False)
    mapping = json.loads(json.dumps(MAPPING_EXAMPLE))
    mapping["channel_aliases"].update({"meta_ads": "META_ADS", "google_ads": "GOOGLE_ADS", "marketplace": "MARKETPLACE"})
    mapping["date_format"], mapping["outcome_date_format"] = "%d/%m/%Y %H:%M", "%d/%m/%Y"
    df, rep = load_merchant_orders(csv, mapping)
    assert rep["rows_in"] == 600 and rep["rows_out"] == 600 and rep["dropped"]["unparseable_date"] == 0
    assert "weight_grams" in rep["defaults_used"] and "device_fingerprint" not in rep["defaults_used"]
    merged = df.set_index("order_id").join(src.set_index("order_id"), rsuffix="_src")
    assert (merged.rto == merged.rto_src).all()                                # label parity
    assert (merged.customer_phone == merged.customer_phone_src).all()          # phone normalised back to 10 digits
    assert (merged.payment_method == merged.payment_method_src).all()
    assert (merged.acquisition_channel == merged.acquisition_channel_src).all()
    assert abs(merged.ts - merged.ts_src).max() < 60                          # minute-resolution export
    assert df.pin_tier.between(1, 4).all() and df.is_new_customer.iloc[0]


def test_missing_required_column_is_explicit(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"Order ID": ["a"], "Order Date": ["01/01/2025"]}).to_csv(csv, index=False)
    try:
        load_merchant_orders(csv, MAPPING_EXAMPLE)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "required columns" in str(e) and "customer_phone" in str(e)
    write_mapping_example(tmp_path / "m.json")
    assert json.loads((tmp_path / "m.json").read_text())["columns"]["order_id"] == "Order ID"


def test_date_format_list_parses_mixed_columns(tmp_path):
    """A real export mixed MM-DD-YY and MM-DD-YYYY in one column; every row must keep the first format that parses."""
    import json
    import pandas as pd
    from chakrashield.data.adapter import load_merchant_orders, write_mapping_example

    mp = tmp_path / "m.json"
    write_mapping_example(mp)
    mapping = json.loads(mp.read_text(encoding="utf-8"))
    mapping["date_format"] = ["%m-%d-%y", "%m-%d-%Y"]
    cols = mapping["columns"]
    rows = [{cols["order_id"]: f"o{i}", cols["ts"]: d, cols["customer_phone"]: f"98765{i:05d}", cols["shipping_address"]: "12, MG Road, Pune",
             cols["delivery_pin"]: "411001", cols["cart_gmv"]: 500 + i, cols["payment_method"]: "COD", cols["status"]: "Delivered"}
            for i, d in enumerate(["04-30-22", "05-03-2022", "06-29-22", "03-31-2022"])]
    csv = tmp_path / "x.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df, rep = load_merchant_orders(csv, mapping)
    assert rep["dropped"]["unparseable_date"] == 0 and len(df) == 4
    assert sorted(pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d")) == ["2022-03-31", "2022-04-30", "2022-05-03", "2022-06-29"]
