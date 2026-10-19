"""Load transactions, model predictions and model metrics into SQL.

Default is a local SQLite file (no setup needed). For MySQL set:
  export DB_URL="mysql+pymysql://USER:PASSWORD@localhost:3306/fraud_db"
(create the empty database first:  CREATE DATABASE fraud_db;)
Also exports CSVs in reports/powerbi/ for the Power BI dashboard.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from features import add_features, load_raw


def main(args):
    root = Path(args.root)
    engine = create_engine(os.getenv("DB_URL", f"sqlite:///{root / 'data/fraud.db'}"))

    df = add_features(load_raw(args.data))
    txns = pd.DataFrame({"txn_id": df["txn_id"], "time_seconds": df["Time"],
                         "hour_of_day": df["hour"], "amount": df["Amount"],
                         "is_fraud": df["Class"]})
    preds = pd.read_csv(root / "reports/test_predictions.csv")
    preds = preds[["txn_id", "fraud_score", "predicted_fraud", "model_name", "threshold"]]
    mm = pd.read_csv(root / "reports/model_comparison.csv").rename(columns={
        "model": "model_name", "test_precision@0.5": "test_precision_05",
        "test_recall@0.5": "test_recall_05", "test_f1@0.5": "test_f1_05"})

    with engine.begin() as conn:
        for stmt in (root / "sql/schema.sql").read_text().split(";"):
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            if "".join(lines).strip():
                conn.execute(text("\n".join(lines)))
        txns.to_sql("transactions", conn, if_exists="append", index=False, chunksize=5000)
        preds.to_sql("predictions", conn, if_exists="append", index=False, chunksize=5000)
        mm.to_sql("model_metrics", conn, if_exists="append", index=False)
    print(f"Loaded {len(txns):,} transactions, {len(preds):,} predictions, {len(mm)} model rows")

    pbi = root / "reports/powerbi"
    pbi.mkdir(parents=True, exist_ok=True)
    txns.merge(preds, on="txn_id", how="left").to_csv(pbi / "transactions_scored.csv", index=False)
    mm.to_csv(pbi / "model_metrics.csv", index=False)
    print(f"Power BI CSVs written to {pbi}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--root", default=".")
    main(ap.parse_args())
