"""Apply the Food-11 two-standard-error model promotion gate using MLflow aliases."""

from __future__ import annotations

import argparse
import json
import math
import os

from mlflow import MlflowClient

MODEL_NAME = "food11"


class GateError(ValueError):
    """The gate cannot safely evaluate the supplied registry/run metadata."""


def evaluation_metadata(run) -> tuple[float, int, str]:
    if run.info.status != "FINISHED":
        raise GateError("Both originating runs must be FINISHED")
    accuracy = run.data.metrics.get("test_accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (float, int)):
        raise GateError("Each originating run must contain test_accuracy")
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise GateError("test_accuracy must be finite and within [0, 1]")
    dataset = run.data.params.get("dataset")
    if dataset not in {"mini", "processed"}:
        raise GateError("Missing or unsupported dataset identity")
    try:
        n = int(run.data.params["evaluation_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError("Missing or invalid evaluation_samples; never infer n from a different dataset") from error
    if n <= 0:
        raise GateError("evaluation_samples must be positive")
    return float(accuracy), n, dataset


def promote_version(client, version: str) -> dict:
    """Evaluate all evidence before changing an alias; report an already-champion no-op."""
    if not str(version).isdigit() or int(version) <= 0:
        raise GateError("Model version must be a positive integer")
    version = str(int(version))
    candidate = client.get_model_version(MODEL_NAME, version)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if not candidate.run_id or not champion.run_id:
        raise GateError("Both model versions must reference their originating runs")
    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)
    acc_candidate, candidate_n, candidate_dataset = evaluation_metadata(candidate_run)
    acc_champion, n, dataset = evaluation_metadata(champion_run)
    if (candidate_n, candidate_dataset) != (n, dataset):
        raise GateError("Candidate and champion must use the same evaluation dataset and sample count")
    # If provenance hashes were logged, require agreement rather than ignoring them.
    candidate_hash = candidate_run.data.params.get("evaluation_sha256")
    champion_hash = champion_run.data.params.get("evaluation_sha256")
    if candidate_hash and champion_hash and candidate_hash != champion_hash:
        raise GateError("Evaluation dataset hashes differ")
    se = math.sqrt(acc_champion * (1 - acc_champion) / n)
    threshold = acc_champion + 2 * se
    decision = ("already_champion" if str(champion.version) == version else
                "promoted" if acc_candidate >= threshold else "challenger")
    report = {
        "candidate_version": version, "candidate_run_id": candidate.run_id,
        "champion_before": str(champion.version), "champion_run_id": champion.run_id,
        "candidate_accuracy": acc_candidate, "champion_accuracy": acc_champion,
        "dataset": dataset, "n": n, "se": se, "threshold": threshold, "decision": decision,
    }
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if str(current.version) != str(champion.version):
        raise GateError("Champion changed during evaluation; retry against the new champion")
    for key, value in report.items():
        client.set_model_version_tag(MODEL_NAME, version, f"quality_gate.{key}", str(value))
    if decision != "already_champion":
        client.set_registered_model_alias(
            MODEL_NAME, "champion" if decision == "promoted" else "challenger", version
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        client = MlflowClient(tracking_uri=uri)
        report = promote_version(client, args.version)
        print(json.dumps(report, indent=2))
        print("Aliases:", json.dumps(client.get_registered_model(MODEL_NAME).aliases, sort_keys=True))
    except GateError as error:
        print(f"FAIL quality gate: {error}. No promotion performed.")
        return 1
    except Exception as error:
        # Do not expose credentials embedded in connection strings or HTTP exceptions.
        print(f"FAIL MLflow operation ({type(error).__name__}); check connectivity, version, champion alias, "
              "and permissions. Inspect alias state before retrying.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
