"""
main.py — Fraud Detection API  v5.0
Adds rich OpenAPI documentation on top of v4.
Every addition here shows up in /docs — zero extra libraries needed.
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from pathlib import Path

# Constants — read from environment variables
# In Docker: set by ENV in Dockerfile or -e flag at runtime.
# Locally:   fall back to the defaults in the second argument of os.getenv().
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH      = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "fraud_model.pkl"))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

_prediction_cache: dict = {}
_cache_hits   = 0
_cache_misses = 0
MAX_CACHE_SIZE = 1000


def make_cache_key(amount, time_of_day, merchant_risk):
    return (round(amount, 2), round(time_of_day, 2), round(merchant_risk, 2))


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    print("[startup] Model ready!")
    yield
    del app.state.model


# OPENAPI ADDITION 1: App-level metadata
#
#   These fields appear on the /docs landing page:
#   - title        → the big H1 heading
#   - summary      → one-line description under the title
#   - description  → Markdown paragraph (supports **bold**, `code`, links)
#   - version      → shown next to the title
#   - contact      → shown at the bottom of the /docs page
#   - license_info → shown at the bottom of the /docs page

app = FastAPI(
    title="Fraud Detection API",
    summary="Real-time credit card fraud prediction using XGBoost.",
    description="""
## Fraud Detection API

Scores credit card transactions in real time using a trained **XGBoost** model.

### How it works
1. Send a `POST /predict` request with transaction features
2. The model returns a fraud probability between `0.0` and `1.0`
3. Transactions with probability ≥ 0.5 are flagged for review

### Features used
| Feature | Range | Meaning |
|---------|-------|---------|
| `amount` | 0 – 1,000,000 | Transaction amount in USD |
| `time_of_day` | 0.0 – 23.99 | Hour of day (0 = midnight) |
| `merchant_risk` | 0.0 – 1.0 | Pre-computed merchant risk score |

### Authentication
No authentication required for this demo API.
""",
    version="5.0.0",
    contact={
        "name": "Fraud ML Team",
        "email": "fraud-api@example.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)


# OPENAPI ADDITION 2: Tags — group endpoints into sections
#
#   Tags are labels that group endpoints together in /docs.
#   Every endpoint gets a tag. Related endpoints appear under the same header.
#   The openapi_tags list defines display order and adds descriptions.

openapi_tags = [
    {
        "name": "predictions",
        "description": "Submit transactions and receive fraud predictions.",
    },
    {
        "name": "monitoring",
        "description": "Server health and cache performance statistics.",
    },
]
app.openapi_tags = openapi_tags


# Timing middleware (unchanged)
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed:.2f}"
    return response


# OPENAPI ADDITION 3: Field-level examples
#
#   Adding `examples` to Field() makes /docs show a pre-filled request body
#   in the "Try it out" panel. The user clicks Execute without typing anything.
#
#   We provide two named examples: a clean transaction and a suspicious one.

class Transaction(BaseModel):
    amount: float = Field(
        gt=0,
        le=1_000_000,
        description="Transaction amount in USD — must be greater than $0.01",
        examples=[34.99, 8500.00],
    )
    time_of_day: float = Field(
        ge=0.0,
        le=23.99,
        description="Hour of the transaction. 0.0 = midnight, 23.99 = 11:59 PM",
        examples=[13.5, 2.3],
    )
    merchant_risk: float = Field(
        ge=0.0,
        le=1.0,
        description="Pre-computed merchant risk score. 0.0 = very safe, 1.0 = very high risk",
        examples=[0.1, 0.92],
    )

    @field_validator("amount")
    @classmethod
    def round_amount_to_cents(cls, v):
        if v < 0.01:
            raise ValueError("Amount must be at least $0.01")
        return round(v, 2)

    # OPENAPI ADDITION 4: model_config with schema examples
    #
    #   model_config lets us inject full named examples into the JSON schema.
    #   These appear as the "Example Value" dropdown in /docs.

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Clean transaction — everyday purchase",
                    "value": {
                        "amount": 34.99,
                        "time_of_day": 13.5,
                        "merchant_risk": 0.1,
                    },
                },
                {
                    "summary": "Suspicious transaction — likely fraud",
                    "value": {
                        "amount": 8500.00,
                        "time_of_day": 2.3,
                        "merchant_risk": 0.92,
                    },
                },
            ]
        }
    }


# OPENAPI ADDITION 5: Response model with field descriptions
class Prediction(BaseModel):
    is_fraud: bool = Field(
        description="True if the transaction is predicted to be fraudulent"
    )
    fraud_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score — probability of fraud between 0.0 and 1.0",
    )
    message: str = Field(
        description="Human-readable verdict: 'Flagged for review' or 'Transaction looks clean'"
    )
    cache_hit: bool = Field(
        default=False,
        description="True if this result was served from the prediction cache",
    )


# Health check — with tag + docstring
@app.get(
    "/",
    tags=["monitoring"],
    summary="Health check",
    response_description="Server status and model state",
)
async def health_check():
    """
    Returns the current health status of the API server.

    - **status**: always `"healthy"` if the server is running
    - **model**: the model version currently loaded in memory
    - **model_loaded**: `true` if the XGBoost model is ready to serve predictions
    """
    return {
        "status": "healthy",
        "model": "XGBoost Fraud Detector v5.0",
        "model_loaded": hasattr(app.state, "model"),
    }


# Stats endpoint — with tag + docstring
@app.get(
    "/stats",
    tags=["monitoring"],
    summary="Cache statistics",
    response_description="Current prediction cache hit/miss counts",
)
async def stats():
    """
    Returns prediction cache performance statistics.

    Use this to understand how often the model is actually running
    versus serving cached results.
    """
    total = _cache_hits + _cache_misses
    return {
        "cache_size"  : len(_prediction_cache),
        "cache_hits"  : _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate_pct": round(_cache_hits / total * 100, 1) if total > 0 else 0.0,
    }


# OPENAPI ADDITION 6: Endpoint-level docs — summary, description, responses ─
#
#   Every decorator argument below adds something to /docs:
#   - summary     → the one-line title shown on the collapsed endpoint card
#   - description → shown when the card is expanded (supports Markdown)
#   - responses   → documents non-200 responses so callers know what to expect
#   - The docstring → becomes the detailed description in /docs

@app.post(
    "/predict",
    tags=["predictions"],
    summary="Predict fraud probability for a transaction",
    response_model=Prediction,
    response_description="Fraud prediction result with probability score",
    responses={
        400: {
            "description": "Transaction exceeds the automated review limit of $500,000",
            "content": {
                "application/json": {
                    "example": {"detail": "Amount $750,000.00 exceeds the automated review limit."}
                }
            },
        },
        422: {
            "description": "Validation error — one or more input fields failed Pydantic validation",
        },
        500: {
            "description": "Unexpected model inference failure",
        },
    },
)
async def predict(transaction: Transaction):
    """
    Scores a credit card transaction and returns a fraud prediction.

    ### Input
    Provide the transaction's **amount**, **time of day**, and **merchant risk score**.

    ### Output
    - `is_fraud`: `true` if fraud probability ≥ 0.5
    - `fraud_probability`: raw model score between 0.0 and 1.0
    - `message`: human-readable verdict
    - `cache_hit`: `true` if the result came from the prediction cache

    ### Notes
    - Transactions above $500,000 are routed to manual review (returns 400)
    - Identical transactions are cached — repeated calls return instantly
    """
    global _cache_hits, _cache_misses

    if transaction.amount > 500_000:
        raise HTTPException(
            status_code=400,
            detail=f"Amount ${transaction.amount:,.2f} exceeds the automated review limit of $500,000.",
        )

    cache_key = make_cache_key(
        transaction.amount, transaction.time_of_day, transaction.merchant_risk
    )
    if cache_key in _prediction_cache:
        _cache_hits += 1
        return Prediction(**_prediction_cache[cache_key], cache_hit=True)

    _cache_misses += 1

    try:
        features    = np.array([
            transaction.amount, transaction.time_of_day, transaction.merchant_risk
        ]).reshape(1, -1)
        probability = float(app.state.model.predict_proba(features)[0][1])
        probability = round(probability, 4)
        is_fraud    = probability >= FRAUD_THRESHOLD
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    result = {
        "is_fraud": is_fraud,
        "fraud_probability": probability,
        "message": "Flagged for review" if is_fraud else "Transaction looks clean",
    }
    if len(_prediction_cache) >= MAX_CACHE_SIZE:
        del _prediction_cache[next(iter(_prediction_cache))]
    _prediction_cache[cache_key] = result
    return Prediction(**result, cache_hit=False)