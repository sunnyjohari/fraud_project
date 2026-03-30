import pandas as pd
import joblib
import json
import os

TEST_PATH   = "data/processed/test.csv"
MODEL_PATH  = "models/fraud_model.pkl"
REPORT_PATH = "reports/metrics.json"
FEATURES    = ["amount", "time_of_day", "merchant_risk"]
LABEL       = "label"


def run():
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score
    )

    test  = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    X = test[FEATURES].values
    y = test[LABEL].values

    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= 0.5).astype(int)

    metrics = {
        "auc":       round(roc_auc_score(y, probs), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall":    round(recall_score(y, preds, zero_division=0), 4),
        "f1":        round(f1_score(y, preds, zero_division=0), 4),
        "test_rows": len(test),
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[evaluate]  AUC={metrics['auc']}  "
          f"precision={metrics['precision']}  "
          f"recall={metrics['recall']}  "
          f"F1={metrics['f1']}")
    print(f"            report → {REPORT_PATH}")


if __name__ == "__main__":
    run()
