# Lab 3 Questions

Verified on 2026-09-27 with the existing Lab 2 model; no retraining or Git commits.

1. **Version and registry:** The first registration was `food11` version **1**.
   A logged model is an experiment output associated with a training run. A
   registered model adds a shared name, numbered versions, aliases, and deployment
   metadata. The original best run is `6ecfb36f2ffd4ed19e49804c3c2ec39a`.
   Its existing MLflow 3 logged-model URI is
   `models:/m-ff17791f2ea2479caf317b0274622353`, not an assumed `runs:/.../model`.
   The publisher copies these original artifacts to an HTTP-accessible deployment
   run, preserving the original run ID/model URI in tags.

2. **Aliases:** User-defined aliases such as `champion` and `challenger` replace
   reliance on fixed stages; they are examples, not automatically assigned stages.
   Registry versioning separates deployment decisions from experiment execution.
   An alias is a movable reference, allowing promotion or rollback without
   renaming/retraining a model or changing application code.

3. **Model URI:** `models:/food11@champion` resolves the deployed version and its
   MLflow flavor/metadata. Loading a bare `.pth` would require manually managing
   architecture, deserialization, paths, and version selection. Register the new
   compatible model, assign `champion`, then restart the API. No image rebuild is
   needed unless the model's interface/dependencies or serving code change.

4. **Build cache:** Dependency installation only depends on `pyproject.toml` and
   `uv.lock`, so editing `serve.py` reuses the expensive dependency layer. Only
   the source-copy layer and subsequent layers need rebuilding. This cache reuse
   was observed in the actual rebuild. `--no-install-project` is added because
   source is intentionally absent from the dependency-only builder layer; uvicorn
   imports `src.food11.serve` directly from `/app` at runtime.

5. **Measured image sizes:** Docker Desktop reported approximately **3.37 GB disk
   usage / 788 MB content** for `food11-api:single-stage`, versus **2.00 GB disk
   usage / 427 MB content** for `food11-api:latest`. This saves about **1.37 GB
   disk usage (41%)**, or **362 MB content (46%)**. These are different Docker
   accounting measures, not interchangeable. `docker history` showed the virtual
   environment as the largest layer (about **1.43 GB uncompressed**). The naive
   image also retains the full Debian/build-tool layers (including a ~619 MB
   layer) and ~60.5 MB uv-install layer. Reproduce the comparison with
   `docker build -f tests/Dockerfile.single-stage -t food11-api:single-stage .`.

6. **Docker ignore:** Without exclusions, build context transfer/scanning can
   become much slower and expose datasets, Git history, local artifacts, or
   secrets to the builder. Image size increases only if those files are actually
   copied into image layers. Merely sending a folder does not inherently break
   this Dockerfile, which uses selective `COPY`. A careless `COPY . .` could
   overwrite the Linux virtual environment with the host's Windows `.venv`,
   breaking executables/dependencies; huge data/cache folders can exhaust space.

7. **Host networking:** In a bridged container, `127.0.0.1` is that container's
   loopback, not Windows. Docker Desktop's `host.docker.internal` resolves to a
   host-reachable address. The host server must listen on an accessible interface
   and permit that Host header; the guide supplies those options.

8. **Fresh container:** Yes. The standalone container was removed and recreated
   from the same image, and it loaded version 1 again. Code/dependencies are in
   the image; model artifacts are fetched from MLflow at startup. The sample
   `validation/Bread/0_0.jpg` returned `Meat` with confidence about `0.478903`.
   This is the existing model's prediction (a misclassification), not a claim of
   accuracy. Windows CPU and Linux CPU differed only by about 0.0000005 in score.

9. **Distributing the exact image:** It still needs to be tagged and pushed to an
   accessible container registry (with appropriate credentials/permissions).
   Consumers should pin its immutable digest rather than a mutable `latest` tag,
   use a compatible architecture, and configure access to MLflow/artifacts.
   Git versions the recipe, not the locally built image. Nothing was pushed here.
