"""
Five tests. Each one tests exactly one thing.

"""


#  Test 1: Health check 
#
#   The simplest test possible.
#   A healthy server should always return 200 with status="healthy".
#   If this test fails — the server can't even start.

def test_health_check(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


#  Test 2: Clean transaction returns a safe prediction 
#
#   A small daytime purchase at a low-risk merchant.
#   The model should predict NOT fraud with very low probability.

def test_clean_transaction_is_not_fraud(client):
    response = client.post("/predict", json={
        "amount"       : 34.99,
        "time_of_day"  : 13.5,
        "merchant_risk": 0.1
    })

    assert response.status_code == 200

    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5    # well below the threshold
    assert data["message"] == "Transaction looks clean"


#  Test 3: Suspicious transaction is flagged as fraud
#
#   Large amount, 2 AM, high-risk merchant — every signal points to fraud.
#   The model should return is_fraud=True.

def test_suspicious_transaction_is_fraud(client):
    response = client.post("/predict", json={
        "amount"       : 8500.00,
        "time_of_day"  : 2.3,
        "merchant_risk": 0.92
    })

    assert response.status_code == 200

    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.5    # above the fraud threshold
    assert data["message"] == "Flagged for review"


#  Test 4: Invalid input returns 422 
#
#   A negative amount is not a valid transaction.
#   Pydantic should reject it before the model ever runs.
#   FastAPI returns 422 Unprocessable Entity automatically.

def test_negative_amount_returns_422(client):
    response = client.post("/predict", json={
        "amount"       : -100.0,    # ← invalid: must be > 0
        "time_of_day"  : 10.0,
        "merchant_risk": 0.3
    })

    assert response.status_code == 422   # Pydantic validation error

    # FastAPI's 422 response always has a "detail" list with error info
    errors = response.json()["detail"]
    assert len(errors) > 0               # at least one validation error reported


#  Test 5: Amount over limit returns 400 
#
#   $750,000 is a valid float that passes Pydantic — but our business rule
#   says amounts above $500,000 must go to manual review, not the ML model.
#   The endpoint should raise HTTPException(400).

def test_amount_over_limit_returns_400(client):
    response = client.post("/predict", json={
        "amount"       : 750_000.00,   #  valid float, breaks business rule
        "time_of_day"  : 9.0,
        "merchant_risk": 0.4
    })

    assert response.status_code == 400

    detail = response.json()["detail"]
    assert "500,000" in detail            # the error message mentions the limit
