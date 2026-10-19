"""Data loading and feature engineering shared by all scripts."""
import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["log_amount", "hour"]
REQUIRED = ["Time", "Amount", "Class"] + V_COLS


def load_raw(path):
    """Load the Kaggle Credit Card Fraud CSV, drop exact duplicates, add txn_id."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, "txn_id", df.index + 1)
    print(f"Loaded {n_before:,} rows, dropped {n_before - len(df):,} duplicates -> {len(df):,}")
    return df


def add_features(df):
    """Time is seconds since the first transaction (2 days of data).
    hour = hour of day; log_amount reduces the heavy right skew of Amount."""
    df = df.copy()
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    return df
