import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def frontend(monkeypatch):
    # Streamlit is installed in its own image, not in the training environment.
    st = Mock()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setenv("INFERENCE_URL", "http://inference:8000/")
    spec = importlib.util.spec_from_file_location("frontend_under_test", Path(__file__).parents[1] / "frontend/app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, st


def test_posts_multipart_to_configured_service(frontend, monkeypatch):
    app, _ = frontend
    response = Mock()
    response.json.return_value = {"category": "Bread", "confidence": 0.8}
    post = Mock(return_value=response)
    monkeypatch.setattr(app.requests, "post", post)
    assert app.request_prediction("x.png", b"image", "image/png") == response.json.return_value
    post.assert_called_once_with("http://inference:8000/predict",
                                 files={"file": ("x.png", b"image", "image/png")}, timeout=(5, 60))
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("result", [{}, {"category": "Bread", "confidence": float("nan")},
                                   {"category": "Bread", "confidence": 2}])
def test_invalid_prediction_schema(frontend, monkeypatch, result):
    app, _ = frontend
    response = Mock()
    response.json.return_value = result
    monkeypatch.setattr(app.requests, "post", Mock(return_value=response))
    with pytest.raises(ValueError, match="category and confidence"):
        app.request_prediction("x.png", b"image", "image/png")


def test_connection_failure_is_shown_to_user(frontend, monkeypatch):
    app, st = frontend
    st.file_uploader.return_value = SimpleNamespace(name="x.png", type="image/png", getvalue=lambda: b"image")
    st.spinner.return_value = io.StringIO()
    monkeypatch.setattr(app, "request_prediction", Mock(side_effect=requests.ConnectionError()))
    app.main()
    st.error.assert_called_once()
    assert "starting or unavailable" in st.error.call_args.args[0]
