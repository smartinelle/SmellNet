from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.models import GCMSMLPEncoder, Transformer

from autoresearch_smellnet.prepare import (
    BaselineConfig,
    accuracy_at_k,
    make_dataloaders,
    prepare_baseline_splits,
    save_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmellNet autoresearch baseline training entrypoint.")
    parser.add_argument("--train-dir", default="data/offline_training")
    parser.add_argument("--test-dir", default="data/offline_testing")
    parser.add_argument("--output-dir", default="autoresearch_runs/baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradient-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--track", default="exact-upstream")
    parser.add_argument("--primary-metric", default="window_acc@1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            logits_all.append(logits.cpu())
            labels_all.append(batch_y.cpu())

    logits_tensor = torch.cat(logits_all, dim=0)
    labels_tensor = torch.cat(labels_all, dim=0)
    return {
        "loss": total_loss / max(total_examples, 1),
        "acc@1": accuracy_at_k(logits_tensor, labels_tensor, 1),
        "acc@5": accuracy_at_k(logits_tensor, labels_tensor, 5),
    }


def train_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc@1": 100.0 * total_correct / max(total_examples, 1),
    }


def build_transformer(
    *,
    input_dim: int,
    num_classes: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> Transformer:
    return Transformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)


def build_gcms_encoder(
    *,
    input_dim: int,
    embedding_dim: int,
    dropout: float,
    device: torch.device,
) -> GCMSMLPEncoder:
    return GCMSMLPEncoder(
        in_features=input_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        l2_normalize=False,
    ).to(device)


def best_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    baseline_config = BaselineConfig(
        gradient_period=args.gradient_period,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    prepared = prepare_baseline_splits(args.train_dir, args.test_dir, baseline_config)
    train_loader, test_loader = make_dataloaders(prepared)

    model = build_transformer(
        input_dim=len(prepared.channel_names),
        model_dim=args.model_dim,
        num_classes=len(prepared.label_names),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, test_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc@1": train_metrics["acc@1"],
            "test_loss": eval_metrics["loss"],
            "test_acc@1": eval_metrics["acc@1"],
            "test_acc@5": eval_metrics["acc@5"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc1={train_metrics['acc@1']:.2f} "
            f"test_acc1={eval_metrics['acc@1']:.2f} "
            f"test_acc5={eval_metrics['acc@5']:.2f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "track": args.track,
        "contrastive": False,
        "architecture": "transformer",
        "input_dim": len(prepared.channel_names),
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_classes": len(prepared.label_names),
    }
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "labels": prepared.label_names,
    }
    torch.save(checkpoint_payload, output_dir / "checkpoint.pt")

    final_metrics = history[-1] if history else {}
    save_run_metadata(
        output_dir,
        prepared=prepared,
        model_config=model_config,
        metrics={
            "track": args.track,
            "contrastive": False,
            "primary_metric": args.primary_metric,
            "device": str(device),
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "history": history,
            "final": final_metrics,
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "final": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
