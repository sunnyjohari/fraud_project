"""
conftest.py — shared pytest fixtures for all test files.

pytest automatically loads this file before running any tests.
The `client` fixture defined here is available to every test
in the tests/ folder — no import needed.
"""

import pytest
from fastapi.testclient import TestClient
from main import app          # import the FastAPI app object from main.py


@pytest.fixture(scope="module")
def client():
    """
    Spin up the full FastAPI app — including the lifespan startup
    (model loading) — and return a TestClient that can make requests.

    scope="module" means: one client is created for the entire test file,
    not a new one for every single test. This avoids reloading the model
    on every test run.

    No real server. No port. No uvicorn.
    TestClient runs the app in-process — fast and fully isolated.
    """
    with TestClient(app) as test_client:
        yield test_client          # tests run here
                                   # lifespan shutdown runs after yield
