"""Publish the existing Lab 2 model to a portable MLflow registry; no retraining."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient


BEST_RUN_ID = "6ecfb36f2ffd4ed19e49804c3c2ec39a"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--source-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-id", default=BEST_RUN_ID)
    parser.add_argument("--new-version", action="store_true", help="Register another version for the Lab 4 alias/restart test")
    args = parser.parse_args()
    if args.source_tracking_uri == "sqlite:///mlflow.db" and not Path("mlflow.db").is_file():
        parser.error("Run this command from the repository root containing Lab 2's mlflow.db")
    source = MlflowClient(tracking_uri=args.source_tracking_uri)
    run = source.get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Only a completed training run can be published")
    model_uri = run.data.tags.get("model_uri")
    if not model_uri or not model_uri.startswith("models:/m-"):
        raise ValueError("Expected Lab 2's logged-model URI tag")
    logged_model = source.get_logged_model(model_uri.removeprefix("models:/"))
    if logged_model.source_run_id != args.run_id:
        raise ValueError("Logged model does not belong to the requested run")
    mlflow.set_tracking_uri(args.tracking_uri)
    target = MlflowClient()
    if not args.new_version:
        for version in target.search_model_versions("name = 'food11'"):
            if (version.tags.get("lab2_run_id") == args.run_id
                    and version.tags.get("lab2_model_uri") == model_uri):
                target.set_registered_model_alias("food11", "champion", version.version)
                print(f"Reused food11 version {version.version}; champion assigned")
                return
    experiment = target.get_experiment_by_name("food11-serving")
    if experiment is None:
        experiment_id = target.create_experiment("food11-serving")
        experiment = target.get_experiment(experiment_id)
    if not experiment.artifact_location.startswith(("mlflow-artifacts:", "http://", "https://")):
        raise ValueError("Start MLflow with --serve-artifacts and --artifacts-destination, then use a new proxied experiment")
    with tempfile.TemporaryDirectory(prefix="food11-publish-") as temporary:
        # Copy existing artifacts through MLflow APIs. Never rewrite Lab 2's records.
        local_model = mlflow.artifacts.download_artifacts(
            artifact_uri=logged_model.artifact_location,
            tracking_uri=args.source_tracking_uri, dst_path=temporary,
        )
        mapping_path = source.download_artifacts(args.run_id, "class_to_idx.json", dst_path=temporary)
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="publish-food11-champion") as published:
            mlflow.set_tags({"purpose": "deployment-copy", "lab2_run_id": args.run_id,
                             "lab2_model_uri": model_uri})
            mlflow.log_params(run.data.params)
            mlflow.log_metrics(run.data.metrics)
            mlflow.log_artifacts(local_model, artifact_path="model")
            mlflow.log_dict(mapping, "class_to_idx.json")
            version = mlflow.register_model(
                f"runs:/{published.info.run_id}/model", "food11",
                tags={"lab2_run_id": args.run_id, "lab2_model_uri": model_uri},
            )
    target.set_registered_model_alias("food11", "champion", version.version)
    print(f"Registered food11 version {version.version}; alias champion")
    print(f"Original training run: {args.run_id}; published run: {version.run_id}")
    print(f"Artifact source: {version.source}")


if __name__ == "__main__":
    main()
