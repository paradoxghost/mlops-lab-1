from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.food11 import serve

FIXTURE = Path(__file__).parent / "fixtures" / "bread.jpg"


class FakeModel:
    def predict(self, batch):
        assert batch.shape == (1, 3, 128, 128) and batch.dtype == np.float32
        # The actual MLflow pyfunc returns logits, not probabilities. These logits
        # yield exactly 0.90 for Bread after the production softmax.
        return np.array([[np.log(90.0)] + [0.0] * 10], dtype=np.float32)


@pytest.fixture
def skipped(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    loader = Mock(side_effect=AssertionError("Must not contact MLflow"))
    monkeypatch.setattr(serve, "load_registered_model", loader)
    monkeypatch.setattr(serve.mlflow, "set_tracking_uri", loader)
    with TestClient(serve.app) as client:
        yield client
    loader.assert_not_called()
    assert serve.model is None


def test_health(skipped):
    response = skipped.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_json(skipped, monkeypatch):
    monkeypatch.setattr(serve, "model", FakeModel())
    with FIXTURE.open("rb") as image:
        response = skipped.post("/predict", files={"file": ("bread.jpg", image, "image/jpeg")})
    assert response.status_code == 200
    assert set(response.json()) == {"category", "confidence"}
    assert response.json()["category"] == "Bread"
    assert 0 <= response.json()["confidence"] <= 1
    assert response.json()["confidence"] == pytest.approx(0.90)


def test_no_model_returns_503(skipped):
    with FIXTURE.open("rb") as image:
        response = skipped.post("/predict", files={"file": ("bread.jpg", image, "image/jpeg")})
    assert response.status_code == 503
    assert response.json()["detail"] == "No model is loaded"
