
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Create synthetic fraud dataset
#    Features:  [amount, time_of_day, merchant_risk]
#    Label:     1 = fraud, 0 = legitimate
np.random.seed(42)
N = 5000

# Legitimate transactions — small amounts, normal hours, low-risk merchants
legit_amount       = np.random.exponential(scale=200,  size=int(N * 0.97))
legit_time         = np.random.uniform(8, 20,           size=int(N * 0.97))
legit_risk         = np.random.uniform(0.0, 0.4,        size=int(N * 0.97))
legit_labels       = np.zeros(int(N * 0.97))

# Fraudulent transactions — large amounts, unusual hours, high-risk merchants
fraud_amount       = np.random.uniform(3000, 15000,     size=int(N * 0.03))
fraud_time         = np.concatenate([
    np.random.uniform(0, 4,  size=int(N * 0.015)),
    np.random.uniform(22, 24, size=int(N * 0.015))
])
fraud_risk         = np.random.uniform(0.6, 1.0,        size=int(N * 0.03))
fraud_labels       = np.ones(int(N * 0.03))

# Stack everything
X = np.vstack([
    np.column_stack([legit_amount, legit_time, legit_risk]),
    np.column_stack([fraud_amount, fraud_time, fraud_risk])
])
y = np.concatenate([legit_labels, fraud_labels])

# 2. Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train XGBoost 
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=32,   # handles class imbalance (97:3 ratio)
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
model.fit(X_train, y_train)

# 4. Quick AUC check
y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
print(f"Model trained successfully — AUC on test set: {auc:.4f}")

# 5. Save model
joblib.dump(model, "fraud_model.pkl")
print("Saved to fraud_model.pkl")
print(f"File size: {__import__('os').path.getsize('fraud_model.pkl') / 1024:.1f} KB")
