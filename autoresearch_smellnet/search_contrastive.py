from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.evaluate import evaluate_contrastive
from models.loss import cross_modal_contrastive_loss
from models.utils import ingredient_to_category

from autoresearch_smellnet.prepare import (
    BaselineConfig,
    DROPPED_SENSOR_COLUMNS,
    PAPER_SENSOR_COLUMNS,
    RAW_SENSOR_COLUMNS,
    load_gcms_bank,
    make_contrastive_train_loader,
    prepare_search_splits,
)
from autoresearch_smellnet.train import (
    best_state_dict,
    build_gcms_encoder,
    build_transformer,
    pick_device,
    set_seed,
)


class _SilentLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budgeted contrastive SmellNet autoresearch loop.")
    parser.add_argument("--train-dir", default="data/offline_training")
    parser.add_argument("--test-dir", default="data/offline_testing")
    parser.add_argument("--gcms-csv", default="data/gcms_dataframe.csv")
    parser.add_argument("--output-dir", default="autoresearch_runs/search_contrastive_exact_upstream")
    parser.add_argument("--time-budget-hours", type=float, default=1.0)
    parser.add_argument("--trial-epochs", type=int, default=20)
    parser.add_argument("--validation-files-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def candidate_stream(rng: random.Random):
    yield {
        "model_dim": 256,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "gcms_dropout": 0.1,
        "lr": 3e-4,
        "temperature": 0.07,
    }
    yield {
        "model_dim": 256,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "gcms_dropout": 0.0,
        "lr": 1e-3,
        "temperature": 0.07,
    }
    yield {
        "model_dim": 512,
        "num_heads": 8,
        "num_layers": 5,
        "dropout": 0.0,
        "gcms_dropout": 0.1,
        "lr": 3e-4,
        "temperature": 0.07,
    }
    head_spaces = {
        128: [4, 8],
        256: [4, 8],
        384: [6, 8],
        512: [8],
    }
    while True:
        model_dim = rng.choice(sorted(head_spaces))
        yield {
            "model_dim": model_dim,
            "num_heads": rng.choice(head_spaces[model_dim]),
            "num_layers": rng.choice([2, 3, 4, 5, 6]),
            "dropout": rng.choice([0.0, 0.05, 0.1, 0.2]),
            "gcms_dropout": rng.choice([0.0, 0.05, 0.1, 0.2]),
            "lr": rng.choice([1e-4, 2e-4, 3e-4, 5e-4, 1e-3]),
            "temperature": rng.choice([0.03, 0.05, 0.07, 0.1, 0.2]),
        }


def metric_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["acc@1"]),
        float(metrics["acc@5"]),
        -float(metrics.get("loss", 0.0)),
    )


def train_contrastive_epoch(
    *,
    gcms_encoder,
    sensor_encoder,
    loader,
    optimizer,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    gcms_encoder.train()
    sensor_encoder.train()
    total_loss = 0.0
    total_batches = 0

    for batch_gcms, batch_sensor in loader:
        batch_gcms = batch_gcms.to(device=device, dtype=torch.float32)
        batch_sensor = batch_sensor.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        z_gcms = gcms_encoder.forward_features(batch_gcms)
        z_sensor = sensor_encoder.forward_features(batch_sensor)
        loss = cross_modal_contrastive_loss(z_gcms, z_sensor, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gcms_encoder.parameters()) + list(sensor_encoder.parameters()),
            1.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return {"loss": total_loss / max(total_batches, 1)}


def build_eval_loader(X: torch.Tensor | None, y: torch.Tensor | None, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def save_artifacts(
    run_dir: Path,
    *,
    prepared,
    gcms_bank,
    metrics: dict,
    candidate: dict,
    sensor_state_dict: dict,
    gcms_state_dict: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sensor_config = {
        "architecture": "transformer",
        "input_dim": len(prepared.channel_names),
        "model_dim": candidate["model_dim"],
        "num_heads": candidate["num_heads"],
        "num_layers": candidate["num_layers"],
        "dropout": candidate["dropout"],
    }
    gcms_config = {
        "architecture": "gcms_mlp_encoder",
        "input_dim": int(gcms_bank.X_gcms.shape[1]),
        "embedding_dim": candidate["model_dim"],
        "dropout": candidate["gcms_dropout"],
    }
    payload = {
        "sensor_encoder_state_dict": sensor_state_dict,
        "gcms_encoder_state_dict": gcms_state_dict,
        "sensor_config": sensor_config,
        "gcms_config": gcms_config,
        "labels": prepared.label_names,
    }
    torch.save(payload, run_dir / "checkpoint.pt")
    preprocessing = {
        "track": "exact-upstream-contrastive",
        "raw_sensor_columns": RAW_SENSOR_COLUMNS,
        "used_sensor_columns": list(PAPER_SENSOR_COLUMNS),
        "dropped_sensor_columns": DROPPED_SENSOR_COLUMNS,
        "gradient_period": prepared.config.gradient_period,
        "window_size": prepared.config.window_size,
        "stride": prepared.config.stride,
        "sensor_scaler_mean": prepared.scaler_mean,
        "sensor_scaler_scale": prepared.scaler_scale,
        "gcms_scaler_mean": gcms_bank.scaler_mean,
        "gcms_scaler_scale": gcms_bank.scaler_scale,
        "labels": prepared.label_names,
        "split_summary": prepared.split_summary,
        "contrastive": {
            "temperature": candidate["temperature"],
            "lr": candidate["lr"],
        },
        "sensor_model": sensor_config,
        "gcms_model": gcms_config,
    }
    (run_dir / "labels.json").write_text(json.dumps({"labels": prepared.label_names}, indent=2) + "\n")
    (run_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    silent_logger = _SilentLogger()

    config = BaselineConfig(
        gradient_period=args.gradient_period,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    prepared = prepare_search_splits(
        args.train_dir,
        args.test_dir,
        config,
        validation_files_per_class=args.validation_files_per_class,
    )
    gcms_bank = load_gcms_bank(args.gcms_csv, expected_labels=prepared.label_names)
    contrastive_train_loader = make_contrastive_train_loader(prepared, gcms_bank)
    val_loader = build_eval_loader(prepared.X_val, prepared.y_val, prepared.config.batch_size)
    test_loader = build_eval_loader(prepared.X_test, prepared.y_test, prepared.config.batch_size)

    rng = random.Random(args.seed)
    total_budget_seconds = max(args.time_budget_hours, 0.0) * 3600.0
    started_at = time.time()
    trials: list[dict] = []
    best_trial = None

    print(
        json.dumps(
            {
                "event": "contrastive_search_start",
                "device": str(device),
                "time_budget_hours": args.time_budget_hours,
                "trial_epochs": args.trial_epochs,
                "split_summary": prepared.split_summary,
            }
        ),
        flush=True,
    )

    for trial_index, candidate in enumerate(candidate_stream(rng), start=1):
        if total_budget_seconds > 0 and (time.time() - started_at) >= total_budget_seconds:
            break

        run_dir = output_dir / f"trial_{trial_index:03d}"
        trial_seed = args.seed + trial_index - 1
        set_seed(trial_seed)

        sensor_encoder = build_transformer(
            input_dim=len(prepared.channel_names),
            num_classes=len(prepared.label_names),
            model_dim=candidate["model_dim"],
            num_heads=candidate["num_heads"],
            num_layers=candidate["num_layers"],
            dropout=candidate["dropout"],
            device=device,
        )
        gcms_encoder = build_gcms_encoder(
            input_dim=gcms_bank.X_gcms.shape[1],
            embedding_dim=candidate["model_dim"],
            dropout=candidate["gcms_dropout"],
            device=device,
        )
        optimizer = torch.optim.Adam(
            list(sensor_encoder.parameters()) + list(gcms_encoder.parameters()),
            lr=candidate["lr"],
        )

        history: list[dict] = []
        best_val = None
        best_epoch = 0
        best_sensor_weights = None
        best_gcms_weights = None

        for epoch in range(1, args.trial_epochs + 1):
            train_metrics = train_contrastive_epoch(
                gcms_encoder=gcms_encoder,
                sensor_encoder=sensor_encoder,
                loader=contrastive_train_loader,
                optimizer=optimizer,
                temperature=float(candidate["temperature"]),
                device=device,
            )
            val_metrics = evaluate_contrastive(
                gcms_encoder=gcms_encoder,
                sensor_encoder=sensor_encoder,
                gcms_data=gcms_bank.X_gcms,
                sensor_data=prepared.X_val,
                sensor_labels=prepared.y_val,
                batch_size=prepared.config.batch_size,
                ingredient_to_category=ingredient_to_category,
                class_names=prepared.label_names,
                device=device,
                logger=silent_logger,
            )
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_acc@1": val_metrics["acc@1"],
                "val_acc@5": val_metrics["acc@5"],
                "val_loss": val_metrics.get("loss"),
            }
            history.append(epoch_metrics)
            print(
                json.dumps(
                    {
                        "event": "contrastive_epoch",
                        "trial": trial_index,
                        "epoch": epoch,
                        "candidate": candidate,
                        "train_loss": round(train_metrics["loss"], 4),
                        "val_acc@1": round(val_metrics["acc@1"], 4),
                        "val_acc@5": round(val_metrics["acc@5"], 4),
                    }
                ),
                flush=True,
            )
            if best_val is None or metric_key(val_metrics) > metric_key(best_val):
                best_val = val_metrics
                best_epoch = epoch
                best_sensor_weights = best_state_dict(sensor_encoder)
                best_gcms_weights = best_state_dict(gcms_encoder)

        assert best_val is not None
        assert best_sensor_weights is not None
        assert best_gcms_weights is not None

        record = {
            "trial": trial_index,
            "seed": trial_seed,
            "candidate": candidate,
            "best_epoch": best_epoch,
            "validation": {
                "acc@1": float(best_val["acc@1"]),
                "acc@5": float(best_val["acc@5"]),
            },
            "history": history,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        save_artifacts(
            run_dir,
            prepared=prepared,
            gcms_bank=gcms_bank,
            metrics=record,
            candidate=candidate,
            sensor_state_dict=best_sensor_weights,
            gcms_state_dict=best_gcms_weights,
        )
        trials.append(record)
        if best_trial is None or metric_key(record["validation"]) > metric_key(best_trial["validation"]):
            best_trial = record

        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        del sensor_encoder
        del gcms_encoder
        gc.collect()

    if best_trial is None:
        raise RuntimeError("No contrastive trials completed within the requested time budget.")

    best_dir = output_dir / f"trial_{best_trial['trial']:03d}"
    checkpoint = torch.load(best_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    final_sensor_encoder = build_transformer(
        input_dim=len(prepared.channel_names),
        num_classes=len(prepared.label_names),
        model_dim=checkpoint["sensor_config"]["model_dim"],
        num_heads=checkpoint["sensor_config"]["num_heads"],
        num_layers=checkpoint["sensor_config"]["num_layers"],
        dropout=checkpoint["sensor_config"]["dropout"],
        device=device,
    )
    final_gcms_encoder = build_gcms_encoder(
        input_dim=gcms_bank.X_gcms.shape[1],
        embedding_dim=checkpoint["gcms_config"]["embedding_dim"],
        dropout=checkpoint["gcms_config"]["dropout"],
        device=device,
    )
    final_sensor_encoder.load_state_dict(checkpoint["sensor_encoder_state_dict"])
    final_gcms_encoder.load_state_dict(checkpoint["gcms_encoder_state_dict"])

    final_test = evaluate_contrastive(
        gcms_encoder=final_gcms_encoder,
        sensor_encoder=final_sensor_encoder,
        gcms_data=gcms_bank.X_gcms,
        sensor_data=prepared.X_test,
        sensor_labels=prepared.y_test,
        batch_size=prepared.config.batch_size,
        ingredient_to_category=ingredient_to_category,
        class_names=prepared.label_names,
        device=device,
        logger=silent_logger,
    )

    summary = {
        "track": "exact-upstream-contrastive-search",
        "device": str(device),
        "time_budget_hours": args.time_budget_hours,
        "trial_epochs": args.trial_epochs,
        "validation_files_per_class": args.validation_files_per_class,
        "trial_count": len(trials),
        "best_trial": best_trial,
        "final_test": {
            "acc@1": float(final_test["acc@1"]),
            "acc@5": float(final_test["acc@5"]),
        },
    }
    (output_dir / "best_trial.json").write_text(json.dumps(best_trial, indent=2) + "\n")
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "acc@1": float(final_test["acc@1"]),
                "acc@5": float(final_test["acc@5"]),
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
