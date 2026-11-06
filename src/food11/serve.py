"""Serve the registered Food-11 champion using its MLflow pyfunc interface."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .data import CLASSES

MODEL_URI = "models:/food11@champion"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOGGER = logging.getLogger(__name__)
model = None


@dataclass
class LoadedModel:
    model: mlflow.pyfunc.PyFuncModel
    categories: list[str]
    version: str


class Prediction(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


def load_registered_model() -> LoadedModel:
    """Fetch the alias and the source run's actual ImageFolder class mapping."""
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias("food11", "champion")
    if not version.run_id:
        raise ValueError("The champion must reference a run containing class_to_idx.json")
    model = mlflow.pyfunc.load_model(MODEL_URI, model_config={"device": "cpu"})
    signature = model.metadata.signature
    if signature is None:
        raise ValueError("The champion is missing its tensor signature")
    inputs, outputs = signature.inputs.to_dict(), signature.outputs.to_dict()
    spec = inputs[0].get("tensor-spec", {}) if len(inputs) == 1 else {}
    if spec.get("dtype") != "float32" or list(spec.get("shape", [])) != [-1, 3, 128, 128]:
        raise ValueError("Expected the Lab 2 float32 NCHW input signature")
    spec = outputs[0].get("tensor-spec", {}) if len(outputs) == 1 else {}
    if list(spec.get("shape", [])) != [-1, 11]:
        raise ValueError("Expected 11 logits from the Lab 2 model")
    mapping_path = client.download_artifacts(version.run_id, "class_to_idx.json")
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if (len(mapping) != 11 or any(type(i) is not int for i in mapping.values())
            or set(mapping.values()) != set(range(11))):
        raise ValueError("class_to_idx.json must contain exactly the indices 0 through 10")
    categories = sorted(mapping, key=mapping.get)
    return LoadedModel(model=model, categories=categories, version=str(version.version))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = None
    if os.environ.get("SKIP_MODEL"):
        app.state.loaded = LoadedModel(model=None, categories=CLASSES.copy(), version="unloaded")
        try:
            yield
        finally:
            model = None
            del app.state.loaded
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    torch.set_num_threads(4)
    timeout = float(os.environ.get("MODEL_LOAD_TIMEOUT_SECONDS", "300"))
    delay = float(os.environ.get("MODEL_LOAD_RETRY_SECONDS", "5"))
    if timeout <= 0 or delay < 0:
        raise ValueError("Model load timeout must be positive and retry delay nonnegative")
    deadline = time.monotonic() + timeout
    while True:
        try:
            app.state.loaded = await asyncio.to_thread(load_registered_model)
            model = app.state.loaded.model
            break
        except Exception as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Could not load models:/food11@champion. Check MLflow readiness, "
                    "alias registration, and HTTP artifact access."
                ) from error
            LOGGER.warning("Champion not ready (%s); retrying in %.1fs", type(error).__name__, delay)
            await asyncio.sleep(delay)
    LOGGER.info("Loaded food11 champion version %s on CPU", app.state.loaded.version)
    try:
        yield
    finally:
        model = None
        del app.state.loaded


app = FastAPI(title="Food-11 classifier", lifespan=lifespan)


def preprocess_image(content: bytes) -> np.ndarray:
    """Match Lab 1 RGB/Lanczos resize and Lab 2 ImageNet normalization."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                rgb = image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
                pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(status_code=400, detail="Upload a valid, readable image") from error
    normalized = (pixels - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


@app.get("/health")
def health() -> dict[str, str]:
    # Uvicorn accepts requests only after lifespan has loaded the model successfully.
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(request: Request, response: Response, file: Annotated[UploadFile, File()]) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail="No model is loaded")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be at most 10 MiB")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    batch = preprocess_image(content)
    loaded = request.app.state.loaded
    try:
        logits = np.asarray(model.predict(batch), dtype=np.float64)
        if logits.shape != (1, 11) or not np.isfinite(logits).all():
            raise ValueError("Expected one finite vector of 11 logits")
        probabilities = np.exp(logits[0] - logits[0].max())
        probabilities /= probabilities.sum()
    except Exception as error:
        LOGGER.error("Model prediction failed (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail="Model prediction failed") from error
    index = int(np.argmax(probabilities))
    response.headers["X-Model-Version"] = loaded.version
    return Prediction(category=loaded.categories[index], confidence=float(probabilities[index]))
