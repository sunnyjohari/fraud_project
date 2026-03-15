import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np

# Constants
MODEL_PATH      = "fraud_model.pkl"
FRAUD_THRESHOLD = 0.5

# OPTIMISATION 3: Prediction cache
#
#   Key idea: if the same transaction arrives twice, don't run the model again.
#   We store results in a plain Python dict.
#   Key  = rounded tuple of (amount, time_of_day, merchant_risk)
#   Value = the Prediction result dict
#
#   We also track hits and misses so we can show stats at /stats

_prediction_cache: dict = {}
_cache_hits   = 0
_cache_misses = 0

MAX_CACHE_SIZE = 1000   # evict oldest when full


def make_cache_key(amount: float, time_of_day: float, merchant_risk: float) -> tuple:
    """Round inputs to 2 decimal places so near-identical values share a cache slot."""
    return (round(amount, 2), round(time_of_day, 2), round(merchant_risk, 2))


# Lifespan — model loaded once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading model from {MODEL_PATH} ...")
    try:
        app.state.model = joblib.load(MODEL_PATH)
        print(f"[startup] Model ready — {type(app.state.model).__name__}")
    except FileNotFoundError:
        raise RuntimeError(f"'{MODEL_PATH}' not found. Run train_and_save_model.py first.")
    yield
    print("[shutdown] Releasing model from memory.")
    del app.state.model


# App
app = FastAPI(
    title="Fraud Detection API",
    description="Optimised XGBoost serving — async + cache + timing",
    version="4.0.0",
    lifespan=lifespan
)


# OPTIMISATION 2: Timing middleware
#
#   Middleware wraps every single request.
#   We record the time before the endpoint runs and after.
#   The difference goes into the response header: X-Response-Time-ms
#
#   This means: open your browser DevTools → Network tab → any request →
#   look at Response Headers → X-Response-Time-ms tells you exactly
#   how long the server spent on that request.

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start    = time.perf_counter()          # start timer
    response = await call_next(request)     # run the actual endpoint
    elapsed  = (time.perf_counter() - start) * 1000   # convert to ms
    response.headers["X-Response-Time-ms"] = f"{elapsed:.2f}"
    return response


# Input / output models
class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000,
                          description="Transaction amount in USD")
    time_of_day: float = Field(ge=0.0, le=23.99,
                               description="Hour — 0.0 midnight, 23.99 11:59 PM")
    merchant_risk: float = Field(ge=0.0, le=1.0,
                                 description="Risk score — 0.0 safe, 1.0 high risk")

    @field_validator("amount")
    @classmethod
    def round_amount_to_cents(cls, v):
        if v < 0.01:
            raise ValueError("Amount must be at least $0.01")
        return round(v, 2)


class Prediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    message: str
    cache_hit: bool = False     # bonus: tell the caller whether we used the cache


# Health check
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost Fraud Detector v4.0",
        "model_loaded": hasattr(app.state, "model")
    }


# Stats endpoint — see cache performance
@app.get("/stats")
async def stats():
    total = _cache_hits + _cache_misses
    return {
        "cache_size"  : len(_prediction_cache),
        "cache_hits"  : _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate_pct": round(_cache_hits / total * 100, 1) if total > 0 else 0.0,
    }


# OPTIMISATION 1: async def
#   Changed from:  def predict(...)
#   Changed to:    async def predict(...)
#
#   async def tells FastAPI: "while this endpoint is waiting for anything
#   (I/O, network, disk), release the event loop so other requests can run."
#
#   For CPU-bound work like model inference it doesn't speed up a single
#   request. But under concurrent load — many requests at once — it lets
#   FastAPI handle them without blocking the queue.

@app.post("/predict", response_model=Prediction)
async def predict(transaction: Transaction):
    global _cache_hits, _cache_misses

    if transaction.amount > 500_000:
        raise HTTPException(status_code=400,
                            detail="Exceeds automated review limit of $500,000.")

    # Check cache first
    cache_key = make_cache_key(
        transaction.amount,
        transaction.time_of_day,
        transaction.merchant_risk
    )

    if cache_key in _prediction_cache:
        _cache_hits += 1
        cached = _prediction_cache[cache_key]
        return Prediction(**cached, cache_hit=True)

    _cache_misses += 1

    # Cache miss, run the real model
    try:
        features    = np.array([
            transaction.amount,
            transaction.time_of_day,
            transaction.merchant_risk
        ]).reshape(1, -1)

        probability = float(app.state.model.predict_proba(features)[0][1])
        probability = round(probability, 4)
        is_fraud    = probability >= FRAUD_THRESHOLD

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Model inference failed: {str(e)}")

    result = {
        "is_fraud"          : is_fraud,
        "fraud_probability" : probability,
        "message"           : "Flagged for review" if is_fraud else "Transaction looks clean",
    }

    # Store in cache (simple eviction: clear when full)
    if len(_prediction_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]

    _prediction_cache[cache_key] = result

    return Prediction(**result, cache_hit=False)
