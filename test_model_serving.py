import requests

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("  FRAUD DETECTION API — REAL MODEL SERVING TEST")
print("=" * 60)

# Test 1: Legitimate transaction
print("\n[T1] Everyday purchase — expect low probability")
r = requests.post(f"{BASE}/predict", json={
    "amount": 34.99,
    "time_of_day": 13.5,
    "merchant_risk": 0.1
})
data = r.json()
print(f"     Status   : {r.status_code}")
print(f"     is_fraud : {data['is_fraud']}")
print(f"     prob     : {data['fraud_probability']}")
print(f"     message  : {data['message']}")

# Test 2: Suspicious — high amount, 2 AM, risky merchant
print("\n[T2] Suspicious transaction — expect high probability")
r = requests.post(f"{BASE}/predict", json={
    "amount": 8500.00,
    "time_of_day": 2.3,
    "merchant_risk": 0.92
})
data = r.json()
print(f"     Status   : {r.status_code}")
print(f"     is_fraud : {data['is_fraud']}")
print(f"     prob     : {data['fraud_probability']}")
print(f"     message  : {data['message']}")

# Test 3: Borderline — medium risk signals
print("\n[T3] Borderline case — model decides")
r = requests.post(f"{BASE}/predict", json={
    "amount": 2500.00,
    "time_of_day": 23.0,
    "merchant_risk": 0.55
})
data = r.json()
print(f"     Status   : {r.status_code}")
print(f"     is_fraud : {data['is_fraud']}")
print(f"     prob     : {data['fraud_probability']}")
print(f"     message  : {data['message']}")

# Test 4: Health check — confirm model_loaded flag
print("\n[T4] Health check — confirm model is loaded")
r = requests.get(f"{BASE}/")
print(f"     {r.json()}")

print("\n" + "=" * 60)
