import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.food11 import serve


CATEGORIES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
              "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]


def image_bytes(mode="RGB"):
    stream = io.BytesIO()
    Image.new(mode, (40, 30), "white").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def loaded(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]], dtype=np.float32)
    bundle = serve.LoadedModel(model, CATEGORIES, "1")
    loader = Mock(return_value=bundle)
    monkeypatch.setattr(serve, "load_registered_model", loader)
    monkeypatch.setenv("MODEL_LOAD_RETRY_SECONDS", "0")
    return bundle, loader


def test_health_predictions_and_single_startup_load(loaded):
    bundle, loader = loaded
    with TestClient(serve.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for _ in range(2):
            response = client.post("/predict", files={"file": ("image.png", image_bytes(), "image/png")})
            assert response.status_code == 200
            assert set(response.json()) == {"category", "confidence"}
            assert response.json()["category"] == "Meat"
            assert response.json()["confidence"] == pytest.approx(np.exp(3) / (10 + np.exp(3)))
            assert response.headers["X-Model-Version"] == "1"
        loader.assert_called_once()
        batch = bundle.model.predict.call_args.args[0]
        assert batch.shape == (1, 3, 128, 128) and batch.dtype == np.float32
        np.testing.assert_allclose(batch[0, :, 0, 0], (1 - serve.MEAN) / serve.STD)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_color_modes(mode):
    assert serve.preprocess_image(image_bytes(mode)).shape == (1, 3, 128, 128)


@pytest.mark.parametrize("content", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_bad_images_are_client_errors(loaded, content):
    with TestClient(serve.app) as client:
        response = client.post("/predict", files={"file": ("bad.png", content)})
        assert response.status_code == 400
        loaded[0].model.predict.assert_not_called()


def test_missing_and_oversized_uploads(loaded, monkeypatch):
    monkeypatch.setattr(serve, "MAX_UPLOAD_BYTES", 4)
    with TestClient(serve.app) as client:
        assert client.post("/predict").status_code == 422
        assert client.post("/predict", files={"file": ("big.png", b"12345")}).status_code == 413


@pytest.mark.parametrize("output", [np.zeros((1, 10)), np.full((1, 11), np.nan)])
def test_invalid_model_output_fails_clearly(loaded, output):
    loaded[0].model.predict.return_value = output
    with TestClient(serve.app) as client:
        response = client.post("/predict", files={"file": ("image.png", image_bytes())})
        assert response.status_code == 503


def test_startup_retries_then_loads_once(loaded):
    bundle, loader = loaded
    loader.side_effect = [ConnectionError("MLflow still starting"), bundle]
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200
        assert loader.call_count == 2


def test_startup_timeout_does_not_report_healthy(loaded, monkeypatch):
    loaded[1].side_effect = ConnectionError("MLflow unavailable")
    monkeypatch.setenv("MODEL_LOAD_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setenv("MODEL_LOAD_RETRY_SECONDS", "0.01")
    with pytest.raises(RuntimeError, match="Could not load"):
        with TestClient(serve.app):
            pass


def test_registry_alias_and_logged_mapping(tmp_path, monkeypatch):
    mapping = tmp_path / "class_to_idx.json"
    mapping.write_text(__import__("json").dumps(dict(zip(CATEGORIES, range(11)))))
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="run-123", version="7")
    client.download_artifacts.return_value = str(mapping)
    model = Mock()
    model.metadata.signature.inputs.to_dict.return_value = [
        {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3, 128, 128]}}]
    model.metadata.signature.outputs.to_dict.return_value = [
        {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 11]}}]
    monkeypatch.setattr(serve.mlflow, "MlflowClient", Mock(return_value=client))
    loader = Mock(return_value=model)
    monkeypatch.setattr(serve.mlflow.pyfunc, "load_model", loader)
    result = serve.load_registered_model()
    loader.assert_called_once_with("models:/food11@champion", model_config={"device": "cpu"})
    client.download_artifacts.assert_called_once_with("run-123", "class_to_idx.json")
    assert result.categories == CATEGORIES and result.version == "7"
