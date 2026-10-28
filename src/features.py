"""Shared feature engineering and preprocessing used by train.py and predict.py."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_fraud"

CATEGORICAL = ["channel", "merchant_category"]

NUMERIC = [
    "log_amount", "amount_to_avg_ratio", "distance_from_home_km",
    "distance_from_last_txn_km", "txns_last_24h", "is_new_device",
    "is_new_merchant", "used_chip", "used_pin", "is_international",
    "card_age_months", "customer_age", "hour", "day_of_week", "is_night",
]

FEATURES = CATEGORICAL + NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns from the raw transaction data."""
    df = df.copy()
    time = pd.to_datetime(df["transaction_time"])
    df["hour"] = time.dt.hour
    df["day_of_week"] = time.dt.dayofweek          # 0 = Monday
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["log_amount"] = np.log1p(df["amount"])      # reduces the effect of very large amounts
    return df


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Turn text columns into numbers (one-hot) and optionally scale numeric columns."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", numeric_step, NUMERIC),
    ])
