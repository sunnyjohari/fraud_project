import pandas as pd
import numpy as np
import os

RAW_PATH  = "data/raw/transactions.csv"
OUT_TRAIN = "data/processed/train.csv"
OUT_TEST  = "data/processed/test.csv"
TEST_SIZE = 0.2
RANDOM_SEED = 42


def run():
    df = pd.read_csv(RAW_PATH)

    # drop rows with missing values
    df = df.dropna()

    # clip outliers: amount capped at 99th percentile
    cap = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(upper=cap)

    # round amount to 2 decimal places (consistent with API input)
    df["amount"] = df["amount"].round(2)

    # train / test split — stratified to preserve fraud ratio
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label"]
    )

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print(f"[preprocess]  train → {OUT_TRAIN}  ({len(train)} rows)")
    print(f"              test  → {OUT_TEST}  ({len(test)} rows)")


if __name__ == "__main__":
    run()
