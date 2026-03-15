"""
benchmark.py — sends 20 requests and compares server-reported timing
between cache misses (model runs) and cache hits (cached result)

Usage:
    python benchmark.py
"""
import requests
import statistics

BASE = "http://127.0.0.1:8000"

# 10 unique transactions (will be cache misses)
unique_transactions = [
    {"amount": round(100 + i * 150, 2), "time_of_day": round(8 + i * 1.5, 2), "merchant_risk": round(0.1 + i * 0.08, 2)}
    for i in range(10)
]

miss_times = []
hit_times  = []

print("Sending 10 unique transactions (cache misses) ...")
for tx in unique_transactions:
    r = requests.post(f"{BASE}/predict", json=tx)
    ms = float(r.headers.get("X-Response-Time-ms", 0))
    miss_times.append(ms)

print("Sending same 10 transactions again (cache hits) ...")
for tx in unique_transactions:
    r = requests.post(f"{BASE}/predict", json=tx)
    ms = float(r.headers.get("X-Response-Time-ms", 0))
    hit_times.append(ms)

print()
print("=" * 48)
print("  BENCHMARK RESULTS (server-side timing)")
print("=" * 48)
print(f"  Cache MISS avg : {statistics.mean(miss_times):.2f} ms")
print(f"  Cache HIT  avg : {statistics.mean(hit_times):.2f} ms")
speedup = statistics.mean(miss_times) / statistics.mean(hit_times)
print(f"  Speedup        : {speedup:.1f}× faster with cache")
print("=" * 48)

r = requests.get(f"{BASE}/stats")
s = r.json()
print(f"\n  Cache size : {s['cache_size']} entries")
print(f"  Hit rate   : {s['hit_rate_pct']}%")
