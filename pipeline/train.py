import pandas as pd
import joblib
import os

TRAIN_PATH  = "data/processed/train.csv"
MODEL_PATH  = "models/fraud_model.pkl"
FEATURES    = ["amount", "time_of_day", "merchant_risk"]
LABEL       = "label"


def run():
    from xgboost import XGBClassifier

    train = pd.read_csv(TRAIN_PATH)
    X = train[FEATURES].values
    y = train[LABEL].values

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=32,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train]  model saved → {MODEL_PATH}")
    print(f"         trained on {len(train)} rows, {len(FEATURES)} features")


if __name__ == "__main__":
    run()
