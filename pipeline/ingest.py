import numpy as np
import pandas as pd
import os

RAW_PATH = "data/raw/transactions.csv"
N = 5000
RANDOM_SEED = 42


def run():
    np.random.seed(RANDOM_SEED)
    n_legit = int(N * 0.97)
    n_fraud = int(N * 0.03)

    legit = pd.DataFrame({
        "amount":        np.random.exponential(200, n_legit),
        "time_of_day":   np.random.uniform(8, 20, n_legit),
        "merchant_risk": np.random.uniform(0.0, 0.4, n_legit),
        "label":         0,
    })

    fraud = pd.DataFrame({
        "amount":        np.random.uniform(3000, 15000, n_fraud),
        "time_of_day":   np.concatenate([
            np.random.uniform(0, 4, n_fraud // 2),
            np.random.uniform(22, 24, n_fraud // 2),
        ]),
        "merchant_risk": np.random.uniform(0.6, 1.0, n_fraud),
        "label":         1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"[ingest]  {len(df)} rows → {RAW_PATH}")
    print(f"          fraud rate: {df['label'].mean():.1%}")


if __name__ == "__main__":
    run()
