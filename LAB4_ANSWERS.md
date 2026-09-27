# Lab 4 Questions

All references to deprecated `Staging` in the handout are interpreted as the
`champion` alias, consistent with Lab 3 and `models:/food11@champion`.

1. **No volume:** Stopping a container retains its writable layer, but removing
   it loses that layer. A newly created standalone MLflow container had only the
   default experiment; the marker experiment created in the previous container
   was gone. No models were registered for this disposable test.

2. **Named volume versus bind mount:** Both can persist data. Docker manages a
   named volume independently of the repository and container lifecycle, avoiding
   hardcoded host paths and host-platform permission/layout coupling. A bind
   mount would work if its directory, permissions, and backups were managed.

3. **Service DNS:** Compose attaches services to a shared network and provides
   service-name DNS. Both `mlflow` and `inference` resolved from the frontend
   container in the test. Unrelated standalone containers do not automatically
   share that Compose service discovery.

4. **Environment variable:** `INFERENCE_URL` separates configuration from code
   and permits the same frontend image to run locally, in Compose, or elsewhere.
   Standalone Docker must explicitly point it at a reachable API; its default
   loopback URL is useful for local Python execution, not another container.

5. **Internal inference port:** The frontend connects to
   `http://inference:8000` on the private Compose network. No host mapping is
   needed. Only the human-facing MLflow UI and frontend are published, reducing
   unnecessary exposure. `EXPOSE`/`expose` documents a port, not a firewall rule.

6. **Readiness:** A startup load can raise a connection/model-not-found error and
   exit if MLflow or `champion` is unavailable. This implementation uses MLflow
   health checks with `condition: service_healthy`, bounded startup retries
   (default 300 seconds, 5-second delay), and `restart: unless-stopped`. Frontend
   startup waits for inference health, which succeeds only after model loading.
   A fresh registry still needs the explicit bootstrap command in the guide;
   retrying cannot invent a missing model. Unit tests cover retry and timeout.

7. **Observed ports:** `docker compose ps` showed MLflow on
   `127.0.0.1:5000->5000/tcp`, frontend on `127.0.0.1:8501->8501/tcp`, and
   inference with only `8000/tcp` (no host mapping). All three were healthy,
   matching the Compose file.

8. **Alias promotion:** Re-registering the same trained model created version
   **2**, and `champion` was assigned to it. A subsequent request still returned
   header `X-Model-Version: 1` until `docker compose restart inference` was run;
   afterward it returned `X-Model-Version: 2`. The weights were intentionally
   identical, so predictions stayed the same; the header proves which version
   was loaded. Refreshing the page alone does not reload an in-memory model.

9. **No rebuild:** Restart creates a new application process and resolves the
   alias again. The image contains serving code and dependencies, not the
   registered weights. A model with a different input interface or incompatible
   dependencies could require code changes and a rebuild.

10. **Persistence:** After the actual main-stack `down`/`up` cycle, `food11`
    version 2, `champion`, and successful predictions all remained. The named
    `mlflow-data` volume holds both SQLite and artifacts. `down -v` deletes the
    named volume and therefore its registry/run/artifact state. This destructive
    case was tested only on a separately named disposable test volume: its marker
    experiment disappeared after deletion/recreation. The real `mlflow-data`
    volume was preserved. An empty registry must be bootstrapped again before
    inference becomes healthy.

11. **Scaling and failover:** Compose can start three local inference replicas
    with `--scale inference=3`, but an explicit load balancer is needed for robust
    request distribution. Compose alone does not provide multi-host scheduling,
    failover, or highly available storage. Production needs an orchestrator,
    health-aware routing, replicated/external database and artifact storage,
    backups, and appropriate authentication/TLS. A single-host SQLite database
    and local named volume do not survive loss of that machine.

## Necessary interpretation of the example Dockerfile

The artifacts still live at `/mlflow-data/mlruns`, as requested, but their client
URI is `mlflow-artifacts:/` and MLflow proxies downloads over HTTP. A literal
client artifact root `/mlflow-data/mlruns` would direct inference to a filesystem
path it cannot access: only MLflow mounts that volume. Existing Windows `file:`
artifact paths also cannot be used directly inside Linux containers, so the
publisher copies the existing model into a new proxied experiment, without
rewriting original Lab 2 records or retraining. See the
[MLflow artifact-serving documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

Readiness conditions follow
[Docker's startup-order documentation](https://docs.docker.com/compose/how-tos/startup-order/).
