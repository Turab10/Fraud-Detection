"""
Train the fraud detection model and save it to models/fraud_model.joblib.

Steps:
  1. Load data/transactions.csv and add features
  2. Split the data: 60% train, 20% validation, 20% test
  3. Train a Gradient Boosting model (the best model in the notebook)
  4. Pick the decision threshold on the validation set
  5. Report results on the test set (data the model has never seen)
  6. Save the model, the threshold and the test results

Usage:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, add_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_DIR = ROOT / "models"
SEED = 42


def split_data(df):
    """60% train, 20% validation, 20% test. Each part keeps the same fraud rate."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    valid, test = train_test_split(temp, test_size=0.5, stratify=temp[TARGET], random_state=SEED)
    return train, valid, test


def build_model():
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),  # trees do not need scaling
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED)),
    ])


def best_threshold(y_true, y_prob):
    """The probability cut-off that gives the best F1 score (balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def main():
    if not DATA_PATH.exists():
        raise SystemExit("data/transactions.csv not found. Run: python data/generate_data.py")

    df = add_features(pd.read_csv(DATA_PATH))
    train, valid, test = split_data(df)
    print(f"Train: {len(train):,} rows | Validation: {len(valid):,} | Test: {len(test):,}")

    model = build_model()
    model.fit(train[FEATURES], train[TARGET])

    threshold = best_threshold(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])

    y_true = test[TARGET]
    y_prob = model.predict_proba(test[FEATURES])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                             "false_negative": int(fn), "true_positive": int(tp)},
    }

    print("\nTest set results")
    print(f"  Decision threshold : {metrics['threshold']:.2f}")
    print(f"  Precision          : {metrics['precision']:.1%}  (of flagged transactions, how many were fraud)")
    print(f"  Recall             : {metrics['recall']:.1%}  (of all fraud, how much was caught)")
    print(f"  F1 score           : {metrics['f1']:.3f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.3f}")
    print(f"  Fraud caught       : {tp} of {tp + fn}")
    print(f"  False alarms       : {fp} of {tn + fp} normal transactions")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES},
                MODEL_DIR / "fraud_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {MODEL_DIR / 'fraud_model.joblib'}")


if __name__ == "__main__":
    main()
