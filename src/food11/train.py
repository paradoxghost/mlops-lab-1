"""Fine-tune ImageNet ResNet18 on Food-11 and track a run in local MLflow."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torchvision
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = "http://127.0.0.1:5000"
DATASETS = {"mini": "food11_processed_mini", "processed": "food11_processed"}
SPLITS = ("training", "validation", "evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def check_tracking_server() -> None:
    try:
        with urlopen(f"{TRACKING_URI}/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"MLflow health check returned {response.status}")
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            f"MLflow is unavailable at {TRACKING_URI}. Start the lab's tracking "
            "server from the repository root before training."
        ) from error


def load_datasets(dataset: str, weights: ResNet18_Weights) -> dict:
    root = REPOSITORY_ROOT / "data" / DATASETS[dataset]
    # Preserve Lab 1's 128x128 resolution; use the pretrained weights' statistics.
    preprocessing = weights.transforms()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocessing.mean, std=preprocessing.std),
    ])
    result = {}
    for split in SPLITS:
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {dataset} dataset split: {folder}")
        result[split] = datasets.ImageFolder(folder, transform=transform)
        if len(result[split].classes) != 11:
            raise ValueError(f"{folder} must contain exactly 11 classes")
    mapping = result["training"].class_to_idx
    for split in SPLITS[1:]:
        if result[split].class_to_idx != mapping:
            raise ValueError(f"Class mapping mismatch between training and {split}")
    return result


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        sample_count += labels.size(0)
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += labels.size(0)
    return loss_sum / sample_count, correct / sample_count


def main() -> None:
    # MLflow prints Unicode run links; Windows redirected consoles may use cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Fixed seeds aid repeatability; no claim of bit-for-bit GPU determinism.
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    image_sets = load_datasets(args.dataset, weights)
    check_tracking_server()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    existed = mlflow.get_experiment_by_name("food11") is not None
    experiment = mlflow.set_experiment("food11")
    print(f"Experiment food11: id={experiment.experiment_id}, existed={existed}", flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            image_sets[split], batch_size=args.batch_size,
            shuffle=(split == "training"), num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            generator=generator if split == "training" else None,
        )
        for split in SPLITS
    }
    print(f"Device: {device}; split sizes: "
          f"{ {split: len(ds) for split, ds in image_sets.items()} }", flush=True)
    # Download official pretrained weights if needed; never fall back to random weights.
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 11)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_name = f"{args.dataset}-lr{args.lr:g}-bs{args.batch_size}-seed{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Run ID: {run.info.run_id}", flush=True)
        mlflow.set_tags({"lab": "2", "purpose": "comparison"})
        mlflow.log_params({
            "dataset": args.dataset, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "model": "resnet18", "pretrained": True,
            "weights": weights.name, "optimizer": "Adam", "seed": args.seed,
            "device": device.type, "num_workers": args.num_workers,
            "torch_threads": torch.get_num_threads(), "image_size": 128,
            "num_classes": 11, "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            **{f"{split}_samples": len(ds) for split, ds in image_sets.items()},
        })
        mlflow.log_dict(image_sets["training"].class_to_idx, "class_to_idx.json")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, loaders["training"], criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, loaders["validation"], criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} val_accuracy={val_accuracy:.6f}", flush=True)
        _, test_accuracy = evaluate(model, loaders["evaluation"], criterion, device)
        mlflow.log_metric("test_accuracy", test_accuracy, step=args.epochs)
        print(f"test_accuracy={test_accuracy:.6f}", flush=True)

        # Save a normal PyTorch module for subsequent labs, with a tensor signature.
        model.cpu().eval()
        example = np.zeros((2, 3, 128, 128), dtype=np.float32)
        with torch.no_grad():
            predictions = model(torch.from_numpy(example)).numpy()
        info = mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle",
            signature=infer_signature(example, predictions), input_example=example,
            pip_requirements=[f"torch=={torch.__version__}",
                              f"torchvision=={torchvision.__version__}",
                              f"mlflow=={mlflow.__version__}",
                              f"numpy=={np.__version__}",
                              "--extra-index-url https://download.pytorch.org/whl/cu130"],
        )
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"Logged model: {info.model_uri}", flush=True)
        print(f"Run artifacts: {mlflow.get_artifact_uri()}", flush=True)


if __name__ == "__main__":
    main()
