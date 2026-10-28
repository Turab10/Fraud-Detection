"""
Use the trained model to check new transactions for fraud.

Usage:
    python src/predict.py                                   # uses data/new_transactions.csv
    python src/predict.py --input my_transactions.csv       # your own file
    python src/predict.py --input my.csv --output results.csv

The input file needs the same columns as data/transactions.csv
(the is_fraud column is not needed).
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import add_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"


def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    data = add_features(df)
    prob = bundle["model"].predict_proba(data[bundle["features"]])[:, 1]
    result = df.copy()
    result["fraud_probability"] = prob.round(4)
    result["prediction"] = ["FRAUD" if p >= bundle["threshold"] else "ok" for p in prob]
    return result


def main():
    parser = argparse.ArgumentParser(description="Check transactions for fraud.")
    parser.add_argument("--input", default=str(ROOT / "data" / "new_transactions.csv"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "predictions.csv"))
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run: python src/train.py")

    bundle = joblib.load(MODEL_PATH)
    result = predict(pd.read_csv(args.input), bundle)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    view = result[["transaction_id", "channel", "merchant_category", "amount",
                   "fraud_probability", "prediction"]]
    print(view.to_string(index=False))
    flagged = (result["prediction"] == "FRAUD").sum()
    print(f"\n{flagged} of {len(result)} transactions flagged as fraud "
          f"(threshold {bundle['threshold']:.2f}). Saved to {output}")


if __name__ == "__main__":
    main()
