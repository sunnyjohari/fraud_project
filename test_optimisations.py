import requests
import time

BASE = "http://127.0.0.1:8000"

print("=" * 62)
print("  OPTIMISATION TEST — CACHE + TIMING")
print("=" * 62)

# Test 1: First request — cache MISS, model runs
print("\n[T1] First request — cache MISS (model runs)")
payload = {"amount": 250.00, "time_of_day": 14.0, "merchant_risk": 0.3}
r = requests.post(f"{BASE}/predict", json=payload)
data = r.json()
server_ms = r.headers.get("X-Response-Time-ms", "n/a")
print(f"     cache_hit  : {data['cache_hit']}")
print(f"     is_fraud   : {data['is_fraud']}")
print(f"     probability: {data['fraud_probability']}")
print(f"     ⏱  Server processed in: {server_ms} ms")

# Test 2: Identical request — cache HIT, model skipped
print("\n[T2] Same request again — cache HIT (model skipped)")
r = requests.post(f"{BASE}/predict", json=payload)
data = r.json()
server_ms = r.headers.get("X-Response-Time-ms", "n/a")
print(f"     cache_hit  : {data['cache_hit']}")
print(f"     is_fraud   : {data['is_fraud']}")
print(f"     probability: {data['fraud_probability']}")
print(f"     ⏱  Server processed in: {server_ms} ms  ← much faster!")

# Test 3: Different transaction — cache MISS again
print("\n[T3] Different transaction — cache MISS (model runs)")
payload2 = {"amount": 8500.00, "time_of_day": 2.3, "merchant_risk": 0.9}
r = requests.post(f"{BASE}/predict", json=payload2)
data = r.json()
server_ms = r.headers.get("X-Response-Time-ms", "n/a")
print(f"     cache_hit  : {data['cache_hit']}")
print(f"     is_fraud   : {data['is_fraud']}")
print(f"     probability: {data['fraud_probability']}")
print(f"     ⏱  Server processed in: {server_ms} ms")

# Test 4: Check /stats
print("\n[T4] Cache stats")
r = requests.get(f"{BASE}/stats")
stats = r.json()
print(f"     cache_size  : {stats['cache_size']}")
print(f"     hits        : {stats['cache_hits']}")
print(f"     misses      : {stats['cache_misses']}")
print(f"     hit_rate    : {stats['hit_rate_pct']}%")

print("\n" + "=" * 62)
