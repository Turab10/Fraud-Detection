"""
Generate a synthetic credit card transaction dataset for fraud detection.

Real bank data is private, so this script creates realistic, made-up
transactions. About 1.7% of them are fraud, which is close to what banks
see in real life (fraud is rare).

Fraud is created from four common patterns:
  1. Online fraud      - stolen card details used on shopping sites
  2. Card skimming     - a copied card used in a shop far from home
  3. Account takeover  - someone logs in from a new device and makes many
                         purchases quickly
  4. Hidden fraud      - fraud that looks just like a normal purchase

Some normal customers also behave in unusual ways (travel abroad, big
purchases, late-night shopping), so the model cannot simply memorise
"unusual = fraud".

Usage:
    python data/generate_data.py                 # 100,000 rows (default)
    python data/generate_data.py --rows 50000    # choose the size
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
FRAUD_RATE = 0.017

CATEGORIES = [
    "grocery", "restaurant", "gas_station", "utilities", "health",
    "entertainment", "online_shopping", "electronics", "travel", "jewelry",
]

# Typical (median) purchase amount in rupees for each category
MEDIAN_AMOUNT = {
    "grocery": 800, "restaurant": 600, "gas_station": 1500, "utilities": 2000,
    "health": 1000, "entertainment": 700, "online_shopping": 1200,
    "electronics": 8000, "travel": 12000, "jewelry": 15000,
}

# How often each category appears for normal in-store / online purchases
LEGIT_INSTORE_P = [0.30, 0.20, 0.17, 0.03, 0.10, 0.10, 0.00, 0.05, 0.02, 0.03]
LEGIT_ONLINE_P = [0.08, 0.07, 0.00, 0.20, 0.05, 0.12, 0.35, 0.06, 0.06, 0.01]

# Fraudsters prefer items that are easy to resell
FRAUD_ONLINE_P = [0.02, 0.02, 0.00, 0.02, 0.02, 0.07, 0.30, 0.35, 0.12, 0.08]
FRAUD_INSTORE_P = [0.08, 0.05, 0.10, 0.00, 0.02, 0.05, 0.00, 0.35, 0.05, 0.30]


def _amounts(rng, categories, sigma, boost=1.0):
    medians = np.array([MEDIAN_AMOUNT[c] for c in categories]) * boost
    return np.round(rng.lognormal(np.log(medians), sigma), 2)


def _hours(rng, n, night_share):
    """Mostly daytime hours, with a chosen share of late-night (0-5 am) hours."""
    day = np.mod(np.round(rng.normal(14, 4, n)), 24)
    night = rng.integers(0, 6, n)
    return np.where(rng.random(n) < night_share, night, day).astype(int)


def _block(rng, n, *, online_share, cat_online_p, cat_instore_p, amount_sigma,
           amount_boost, avg_spend_median, night_share, home_km_median,
           far_share, last_km_median, txn24_mean, new_device_p,
           new_merchant_p, chip_p, pin_p, intl_p, card_age_scale):
    """Create n transactions that share the same behaviour settings."""
    online = rng.random(n) < online_share
    cats = np.where(
        online,
        rng.choice(CATEGORIES, n, p=cat_online_p),
        rng.choice(CATEGORIES, n, p=cat_instore_p),
    )
    amount = _amounts(rng, cats, amount_sigma, amount_boost)
    avg_spend = rng.lognormal(np.log(avg_spend_median), 0.5, n)

    home_km = rng.lognormal(np.log(home_km_median), 0.9, n)
    far = rng.random(n) < far_share
    home_km = np.where(far, rng.uniform(150, 2500, n), home_km)
    last_km = rng.lognormal(np.log(last_km_median), 1.0, n)

    in_store = ~online
    return pd.DataFrame({
        "channel": np.where(online, "online", "in_store"),
        "merchant_category": cats,
        "amount": amount,
        "amount_to_avg_ratio": np.round(amount / avg_spend, 3),
        "hour": _hours(rng, n, night_share),
        "distance_from_home_km": np.round(home_km, 1),
        "distance_from_last_txn_km": np.round(last_km, 1),
        "txns_last_24h": rng.poisson(txn24_mean, n) + 1,
        "is_new_device": (online & (rng.random(n) < new_device_p)).astype(int),
        "is_new_merchant": (rng.random(n) < new_merchant_p).astype(int),
        "used_chip": (in_store & (rng.random(n) < chip_p)).astype(int),
        "used_pin": (in_store & (rng.random(n) < pin_p)).astype(int),
        "is_international": (rng.random(n) < intl_p).astype(int),
        "card_age_months": np.clip(
            np.round(rng.gamma(2.0, card_age_scale, n)), 1, 180).astype(int),
        "customer_age": np.clip(
            np.round(rng.normal(40, 12, n)), 18, 80).astype(int),
    })


def generate(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * FRAUD_RATE))
    n_legit = n_rows - n_fraud

    # ---- Normal customers -------------------------------------------------
    normal = dict(
        online_share=0.35, cat_online_p=LEGIT_ONLINE_P,
        cat_instore_p=LEGIT_INSTORE_P, amount_sigma=0.7, amount_boost=1.0,
        avg_spend_median=1500, night_share=0.04, home_km_median=5,
        far_share=0.02, last_km_median=3, txn24_mean=1.5, new_device_p=0.04,
        new_merchant_p=0.20, chip_p=0.88, pin_p=0.60, intl_p=0.02,
        card_age_scale=24,
    )
    # Normal customers who look unusual (travelling, big buys, night shifts)
    unusual = dict(normal, amount_boost=2.5, night_share=0.30, far_share=0.40,
                   last_km_median=40, txn24_mean=3, new_device_p=0.25,
                   new_merchant_p=0.55, chip_p=0.70, pin_p=0.40, intl_p=0.30)

    n_unusual = int(n_legit * 0.05)
    legit = pd.concat([
        _block(rng, n_legit - n_unusual, **normal),
        _block(rng, n_unusual, **unusual),
    ])
    legit["is_fraud"] = 0

    # ---- Fraud patterns ---------------------------------------------------
    shares = {"online": 0.45, "skimming": 0.30, "takeover": 0.15, "hidden": 0.10}
    counts = {k: int(round(v * n_fraud)) for k, v in shares.items()}
    counts["hidden"] = n_fraud - sum(v for k, v in counts.items() if k != "hidden")

    fraud_online = dict(normal, online_share=1.0, cat_online_p=FRAUD_ONLINE_P,
                        amount_sigma=0.8, amount_boost=1.8, night_share=0.30,
                        last_km_median=60, txn24_mean=3, new_device_p=0.55,
                        new_merchant_p=0.75, intl_p=0.30, card_age_scale=18)
    fraud_skim = dict(normal, online_share=0.0, cat_instore_p=FRAUD_INSTORE_P,
                      amount_sigma=0.8, amount_boost=1.6, night_share=0.20,
                      home_km_median=80, far_share=0.35, last_km_median=120,
                      txn24_mean=4, new_merchant_p=0.80, chip_p=0.25,
                      pin_p=0.10, intl_p=0.20, card_age_scale=20)
    fraud_takeover = dict(normal, online_share=1.0, cat_online_p=FRAUD_ONLINE_P,
                          amount_sigma=0.6, amount_boost=1.0, night_share=0.35,
                          last_km_median=200, txn24_mean=7, new_device_p=0.90,
                          new_merchant_p=0.70, intl_p=0.25, card_age_scale=22)

    fraud = pd.concat([
        _block(rng, counts["online"], **fraud_online),
        _block(rng, counts["skimming"], **fraud_skim),
        _block(rng, counts["takeover"], **fraud_takeover),
        _block(rng, counts["hidden"], **normal),
    ])
    fraud["is_fraud"] = 1

    # ---- Combine, add dates and IDs ---------------------------------------
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)

    start = pd.Timestamp("2025-01-01")
    days = rng.integers(0, 181, len(df))  # January to June 2025
    minutes = rng.integers(0, 60, len(df))
    df["transaction_time"] = (
        start + pd.to_timedelta(days, unit="D")
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    df = df.sort_values("transaction_time").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, len(df) + 1)])

    columns = [
        "transaction_id", "transaction_time", "channel", "merchant_category",
        "amount", "amount_to_avg_ratio", "distance_from_home_km",
        "distance_from_last_txn_km", "txns_last_24h", "is_new_device",
        "is_new_merchant", "used_chip", "used_pin", "is_international",
        "card_age_months", "customer_age", "is_fraud",
    ]
    return df[columns]  # 'hour' is dropped; it is rebuilt from transaction_time


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(Path(__file__).parent / "transactions.csv"))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} transactions to {args.out}")
    print(f"Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")

    # A small file of brand-new, unlabelled transactions to try predict.py on.
    # It uses a different seed, so the model has never seen these rows.
    fresh = generate(5_000, seed=args.seed + 1)
    sample = pd.concat([
        fresh[fresh["is_fraud"] == 0].head(8),
        fresh[fresh["is_fraud"] == 1].head(4),
    ]).sort_values("transaction_time")
    sample["transaction_id"] = [f"NEW{i:04d}" for i in range(1, len(sample) + 1)]
    sample_path = Path(args.out).parent / "new_transactions.csv"
    sample.drop(columns="is_fraud").to_csv(sample_path, index=False)
    print(f"Saved {len(sample)} unlabelled example transactions to {sample_path}")


if __name__ == "__main__":
    main()
