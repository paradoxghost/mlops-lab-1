# Labs 3 and 4: verification guide

Use PowerShell with Docker Desktop running in Linux-container mode. Run commands
from `C:\Users\vitor\Downloads\mlops\mlops-lab-1`. No credentials are needed for
these local services. Do not add credentials to Dockerfiles, images, or Git.

## What was implemented and verified

- Existing best Lab 2 model, not a newly trained model: run
  `6ecfb36f2ffd4ed19e49804c3c2ec39a` and logged-model URI
  `models:/m-ff17791f2ea2479caf317b0274622353`.
- Actual pyfunc signature: float32 NumPy `[N, 3, 128, 128]` to 11 logits.
  Uploads are converted to RGB, Lanczos-resized to 128x128, scaled to [0, 1],
  and ImageNet-normalized, matching the existing data/training pipeline.
  Class names are downloaded from that version's run's `class_to_idx.json`.
  Stable softmax produces `{"category": "...", "confidence": 0.0}` (score 0-1).
- Alias-based loading once at startup, with retries and an `X-Model-Version`
  response header for observing promotions. No deprecated stage calls.
- Python 3.13 follows the existing project requirement rather than the handout's
  illustrative Python 3.11. Linux images use CPU PyTorch; existing Windows CUDA
  dependencies remain intact. CPU/CUDA requirement warnings reflect the original
  model's training environment; actual Linux CPU inference passed.
- Both standalone API launches (local Python and Docker), a fresh standalone
  container, all three Compose image builds, real internal HTTP prediction,
  Streamlit AppTest upload/render, alias version 1-to-2 reload, and persistence
  after `down`/`up` passed. The UI displayed `Prediction: Meat (47.9%)` for the
  sample below. This is not a model accuracy evaluation.
- 18 pytest cases passed; compile/import checks, lock consistency, Compose config,
  Git whitespace checks, and DVC status passed. Two upstream test-client
  deprecation warnings remain, with no failing tests.
- The final Compose stack was left healthy: MLflow at http://127.0.0.1:5000 and
  frontend at http://127.0.0.1:8501, with `food11@champion` pointing to version 2.
  The original host MLflow registry separately contains Lab 3 version 1.
- Browser visual appearance was not manually inspected; Streamlit's actual upload
  and rendered result were tested automatically. GPU serving, remote deployments,
  image-registry publishing, load testing, and machine failover were not tested.
- No Git commit/push, retraining, dataset modifications, or DVC pointer changes.

## Lab 3: run these steps in order

### 1. Prepare and start host MLflow (terminal A)

The final Lab 4 stack is already running. Stop it first to free port 5000;
`down` without `-v` preserves its registry volume.

```powershell
Set-Location C:\Users\vitor\Downloads\mlops\mlops-lab-1
docker compose down
python -m uv sync --frozen
python -m uv run mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --default-artifact-root mlflow-artifacts:/ --artifacts-destination ./mlruns --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*"
```

Leave terminal A running. Binding 0.0.0.0 permits Docker Desktop to connect;
this is a local lab server, not an authenticated public deployment.

### 2. Register and run the local API (terminal B)

The helper reuses a matching registration by default; it copies the existing
model only if needed. It preserves source run/model identifiers as tags. It
expects the existing ignored `mlflow.db` and `mlruns` from Lab 2.

```powershell
Set-Location C:\Users\vitor\Downloads\mlops\mlops-lab-1
python -m uv run python -m src.food11.register_model --tracking-uri http://127.0.0.1:5000
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
python -m uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

### 3. Test (terminal C)

Wait for `Application startup complete` in terminal B, then:

```powershell
Set-Location C:\Users\vitor\Downloads\mlops\mlops-lab-1
curl.exe --fail http://127.0.0.1:8000/health
curl.exe --fail -i -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
```

Expected health: `{"status":"ok"}`. Prediction fields are exactly `category`
and `confidence`; the loaded version is a response header, not an extra JSON field.

### 4. Build and run Docker (terminal B)

Stop local uvicorn with Ctrl+C, leaving host MLflow in terminal A running.

```powershell
docker build -t food11-api:latest .
docker run --rm --name food11-api-lab3 -p 127.0.0.1:8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

After startup completes, repeat the two curl commands in terminal C. Stop the
container with Ctrl+C and repeat the same `docker run` and curl commands without
rebuilding to verify fresh-container loading.

For the measured size comparison, stop the API container and run:

```powershell
docker build -f tests/Dockerfile.single-stage -t food11-api:single-stage .
docker image ls food11-api
docker history food11-api:latest
docker history food11-api:single-stage
```

## Lab 4: run these steps in order

### 1. Stop standalone services and start the registry

Stop the standalone API and host MLflow with Ctrl+C in their respective
terminals. The Compose MLflow needs port 5000 to be free.

```powershell
Set-Location C:\Users\vitor\Downloads\mlops\mlops-lab-1
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 120 mlflow
python -m uv run python -m src.food11.register_model --tracking-uri http://127.0.0.1:5000
docker compose up -d --wait --wait-timeout 180
docker compose ps
```

Bootstrap occurs before starting the complete stack so a fresh registry contains
`food11@champion`. MLflow uses `mlflow-data` for both SQLite and artifacts, serving
artifacts over HTTP instead of exposing container-only filesystem paths to clients.

### 2. Verify UI and internal networking

Open http://127.0.0.1:5000 (Models -> food11 -> champion) and
http://127.0.0.1:8501. Upload the sample image used in Lab 3. Inference is not
published on the host in Compose; test it from the frontend container:

```powershell
docker compose cp data/food11_processed_mini/validation/Bread/0_0.jpg frontend:/tmp/food11-test.jpg
docker compose exec -T frontend python -c "import requests; print(requests.get('http://inference:8000/health', timeout=10).json())"
docker compose exec -T frontend python -c "import requests; r=requests.post('http://inference:8000/predict', files={'file':('test.jpg',open('/tmp/food11-test.jpg','rb'),'image/jpeg')}, timeout=60); r.raise_for_status(); print(r.json(), 'version='+r.headers['X-Model-Version'])"
```

### 3. Promote and reload

This intentionally creates another version of the same weights (no retraining).
Version numbers increase on each explicit `--new-version` invocation.

```powershell
python -m uv run python -m src.food11.register_model --tracking-uri http://127.0.0.1:5000 --new-version
```

Repeat the internal POST from step 2: its header still identifies the old loaded
version. Then:

```powershell
docker compose restart inference
docker compose up -d --wait --wait-timeout 180
```

Repeat the internal POST: the header now identifies the new version. Identical
weights mean identical predictions, which is why the header is useful.

### 4. Check persistence and tests

```powershell
docker compose down
docker compose up -d --wait --wait-timeout 180
docker compose ps
python -m uv run python -m pytest -q
python -m uv run python -m py_compile src/food11/serve.py src/food11/register_model.py frontend/app.py
python -m uv lock --check
python -m uv run dvc status
git diff --check
git status --short
```

The registered versions and `champion` should still be visible and uploads should
still work. The temporary test image inside the frontend container must be copied
again if repeating the internal POST after container recreation.

**Do not run `docker compose down -v` on your real stack unless you intend to
delete its MLflow database and artifacts.** Its behavior was already verified
using a separate disposable project/volume containing only test marker data,
then cleaned up. That deleted temporary test data is not recoverable; your
actual `mlflow-data` and original Lab 2 model artifacts were preserved.

To stop the stack safely afterward, use `docker compose down` (without `-v`).
