from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from autoresearch_smellnet.prepare import (
    BaselineConfig,
    DROPPED_SENSOR_COLUMNS,
    PAPER_SENSOR_COLUMNS,
    RAW_SENSOR_COLUMNS,
    make_search_dataloaders,
    prepare_search_splits,
)
from autoresearch_smellnet.train import (
    best_state_dict,
    build_transformer,
    evaluate,
    pick_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 budgeted SmellNet autoresearch loop.")
    parser.add_argument("--train-dir", default="data/offline_training")
    parser.add_argument("--test-dir", default="data/offline_testing")
    parser.add_argument("--output-dir", default="autoresearch_runs/search_phase2_exact_upstream")
    parser.add_argument("--time-budget-hours", type=float, default=1.0)
    parser.add_argument("--stage1-fraction", type=float, default=0.6)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage1-max-trials", type=int, default=None)
    parser.add_argument("--stage2-promote-k", type=int, default=8)
    parser.add_argument("--stage2-fold-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--validation-files-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def candidate_stream(rng: random.Random):
    yield {
        "model_dim": 256,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_ratio": 0.1,
    }
    yield {
        "model_dim": 512,
        "num_heads": 8,
        "num_layers": 5,
        "dropout": 0.0,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_ratio": 0.1,
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
            "dropout": rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
            "lr": rng.choice([1e-4, 2e-4, 3e-4, 5e-4, 8e-4]),
            "weight_decay": rng.choice([0.0, 1e-4, 1e-3, 1e-2]),
            "label_smoothing": rng.choice([0.0, 0.05, 0.1]),
            "warmup_ratio": rng.choice([0.0, 0.1, 0.2]),
        }


def metric_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["acc@1"]),
        float(metrics["acc@5"]),
        -float(metrics["loss"]),
    )


def build_scheduler(optimizer, *, total_epochs: int, warmup_ratio: float):
    warmup_epochs = max(0, min(total_epochs - 1, int(round(total_epochs * warmup_ratio))))

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return epoch_number / warmup_epochs
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch_number - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_epoch_configured(model, loader, optimizer, criterion, device: torch.device) -> dict[str, float]:
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


def _preprocessing_payload(prepared, model_config: dict) -> dict:
    return {
        "track": "exact-upstream-search-phase2",
        "raw_sensor_columns": RAW_SENSOR_COLUMNS,
        "used_sensor_columns": list(PAPER_SENSOR_COLUMNS),
        "dropped_sensor_columns": DROPPED_SENSOR_COLUMNS,
        "gradient_period": prepared.config.gradient_period,
        "window_size": prepared.config.window_size,
        "stride": prepared.config.stride,
        "scaler_mean": prepared.scaler_mean,
        "scaler_scale": prepared.scaler_scale,
        "labels": prepared.label_names,
        "split_summary": prepared.split_summary,
        "model": model_config,
    }


def save_artifacts(run_dir: Path, *, prepared, model_config: dict, metrics: dict, checkpoint_payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, run_dir / "checkpoint.pt")
    (run_dir / "labels.json").write_text(json.dumps({"labels": prepared.label_names}, indent=2) + "\n")
    (run_dir / "preprocessing.json").write_text(json.dumps(_preprocessing_payload(prepared, model_config), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def evaluate_candidate(
    *,
    candidate: dict,
    prepared,
    device: torch.device,
    epochs: int,
    seed: int,
    stage_name: str,
    run_dir: Path,
) -> dict:
    set_seed(seed)
    train_loader, val_loader, _ = make_search_dataloaders(prepared)
    model = build_transformer(
        input_dim=len(prepared.channel_names),
        num_classes=len(prepared.label_names),
        model_dim=candidate["model_dim"],
        num_heads=candidate["num_heads"],
        num_layers=candidate["num_layers"],
        dropout=candidate["dropout"],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["lr"],
        weight_decay=candidate["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_ratio=float(candidate["warmup_ratio"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(candidate["label_smoothing"]))

    history: list[dict] = []
    best_val = None
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch_configured(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc@1": train_metrics["acc@1"],
            "val_loss": val_metrics["loss"],
            "val_acc@1": val_metrics["acc@1"],
            "val_acc@5": val_metrics["acc@5"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        print(
            json.dumps(
                {
                    "event": "phase2_epoch",
                    "stage": stage_name,
                    "epoch": epoch,
                    "candidate": candidate,
                    "train_acc@1": round(train_metrics["acc@1"], 4),
                    "val_acc@1": round(val_metrics["acc@1"], 4),
                    "val_acc@5": round(val_metrics["acc@5"], 4),
                    "lr": round(optimizer.param_groups[0]["lr"], 7),
                }
            ),
            flush=True,
        )
        if best_val is None or metric_key(val_metrics) > metric_key(best_val):
            best_val = val_metrics
            best_epoch = epoch
            best_weights = best_state_dict(model)
        scheduler.step()

    assert best_val is not None
    assert best_weights is not None
    model_config = {
        "track": "exact-upstream-search-phase2",
        "contrastive": False,
        "architecture": "transformer",
        "input_dim": len(prepared.channel_names),
        "model_dim": candidate["model_dim"],
        "num_heads": candidate["num_heads"],
        "num_layers": candidate["num_layers"],
        "dropout": candidate["dropout"],
        "num_classes": len(prepared.label_names),
        "optimizer": {
            "lr": candidate["lr"],
            "weight_decay": candidate["weight_decay"],
            "warmup_ratio": candidate["warmup_ratio"],
            "label_smoothing": candidate["label_smoothing"],
        },
    }
    checkpoint_payload = {
        "model_state_dict": best_weights,
        "model_config": model_config,
        "labels": prepared.label_names,
    }
    metrics = {
        "stage": stage_name,
        "candidate": candidate,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": best_val,
        "history": history,
    }
    save_artifacts(
        run_dir,
        prepared=prepared,
        model_config=model_config,
        metrics=metrics,
        checkpoint_payload=checkpoint_payload,
    )

    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    del model
    gc.collect()

    return {
        "candidate": candidate,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": best_val,
        "history": history,
        "run_dir": str(run_dir),
    }


def average_validation(records: list[dict]) -> dict:
    n = len(records)
    return {
        "loss": sum(r["validation"]["loss"] for r in records) / n,
        "acc@1": sum(r["validation"]["acc@1"] for r in records) / n,
        "acc@5": sum(r["validation"]["acc@5"] for r in records) / n,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_budget_seconds = max(args.time_budget_hours, 0.0) * 3600.0
    started_at = time.time()
    stage1_budget_seconds = total_budget_seconds * float(args.stage1_fraction)

    print(
        json.dumps(
            {
                "event": "phase2_start",
                "device": str(device),
                "time_budget_hours": args.time_budget_hours,
                "stage1_fraction": args.stage1_fraction,
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage2_promote_k": args.stage2_promote_k,
                "stage2_fold_count": args.stage2_fold_count,
            }
        ),
        flush=True,
    )

    base_config = BaselineConfig(
        gradient_period=args.gradient_period,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    prepared_fold0 = prepare_search_splits(
        args.train_dir,
        args.test_dir,
        base_config,
        validation_files_per_class=args.validation_files_per_class,
        validation_fold_offset=0,
    )
    print(
        json.dumps(
            {
                "event": "phase2_split",
                "fold_offset": 0,
                "split_summary": prepared_fold0.split_summary,
            }
        ),
        flush=True,
    )

    rng = random.Random(args.seed)
    stage1_results: list[dict] = []
    for trial_index, candidate in enumerate(candidate_stream(rng), start=1):
        elapsed = time.time() - started_at
        if total_budget_seconds > 0 and elapsed >= stage1_budget_seconds:
            break
        if args.stage1_max_trials is not None and trial_index > args.stage1_max_trials:
            break
        run_dir = output_dir / f"stage1_trial_{trial_index:03d}"
        print(
            json.dumps(
                {
                    "event": "phase2_stage1_trial_start",
                    "trial": trial_index,
                    "candidate": candidate,
                    "elapsed_seconds": round(elapsed, 2),
                }
            ),
            flush=True,
        )
        result = evaluate_candidate(
            candidate=candidate,
            prepared=prepared_fold0,
            device=device,
            epochs=args.stage1_epochs,
            seed=args.seed + trial_index - 1,
            stage_name=f"stage1_trial_{trial_index:03d}",
            run_dir=run_dir,
        )
        result["trial"] = trial_index
        result["elapsed_seconds"] = round(time.time() - started_at, 2)
        stage1_results.append(result)

    ranked_stage1 = sorted(stage1_results, key=lambda r: metric_key(r["validation"]), reverse=True)
    promoted = ranked_stage1[: max(1, min(args.stage2_promote_k, len(ranked_stage1)))]
    (output_dir / "stage1_summary.json").write_text(json.dumps({"results": ranked_stage1, "promoted": promoted}, indent=2) + "\n")

    stage2_candidates: list[dict] = []
    for promoted_index, base_result in enumerate(promoted, start=1):
        if total_budget_seconds > 0 and (time.time() - started_at) >= total_budget_seconds:
            break
        candidate = base_result["candidate"]
        fold_records: list[dict] = []
        for fold_offset in range(args.stage2_fold_count):
            if total_budget_seconds > 0 and (time.time() - started_at) >= total_budget_seconds:
                break
            prepared = prepare_search_splits(
                args.train_dir,
                args.test_dir,
                base_config,
                validation_files_per_class=args.validation_files_per_class,
                validation_fold_offset=fold_offset,
            )
            run_dir = output_dir / f"stage2_candidate_{promoted_index:02d}_fold_{fold_offset:02d}"
            print(
                json.dumps(
                    {
                        "event": "phase2_stage2_fold_start",
                        "candidate_index": promoted_index,
                        "fold_offset": fold_offset,
                        "candidate": candidate,
                    }
                ),
                flush=True,
            )
            record = evaluate_candidate(
                candidate=candidate,
                prepared=prepared,
                device=device,
                epochs=args.stage2_epochs,
                seed=args.seed + 1000 + promoted_index * 100 + fold_offset,
                stage_name=f"stage2_candidate_{promoted_index:02d}_fold_{fold_offset:02d}",
                run_dir=run_dir,
            )
            record["fold_offset"] = fold_offset
            fold_records.append(record)

        if not fold_records:
            continue
        candidate_record = {
            "candidate_index": promoted_index,
            "candidate": candidate,
            "fold_records": fold_records,
            "aggregate_validation": average_validation(fold_records),
        }
        stage2_candidates.append(candidate_record)

    ranked_stage2 = sorted(stage2_candidates, key=lambda r: metric_key(r["aggregate_validation"]), reverse=True)
    best_stage2 = ranked_stage2[0] if ranked_stage2 else None
    (output_dir / "stage2_summary.json").write_text(json.dumps({"results": ranked_stage2}, indent=2) + "\n")

    final_summary = {
        "track": "exact-upstream-search-phase2",
        "device": str(device),
        "time_budget_hours": args.time_budget_hours,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage1_trial_count": len(stage1_results),
        "stage2_candidate_count": len(stage2_candidates),
        "best_stage2": best_stage2,
    }

    if best_stage2 is not None:
        final_candidate = best_stage2["candidate"]
        final_prepared = prepare_search_splits(
            args.train_dir,
            args.test_dir,
            base_config,
            validation_files_per_class=args.validation_files_per_class,
            validation_fold_offset=0,
        )
        final_run = evaluate_candidate(
            candidate=final_candidate,
            prepared=final_prepared,
            device=device,
            epochs=args.stage2_epochs,
            seed=args.seed + 9000,
            stage_name="phase2_final_selection_fold0",
            run_dir=output_dir / "final_selection",
        )
        _, _, test_loader = make_search_dataloaders(final_prepared)
        final_checkpoint = torch.load(Path(final_run["run_dir"]) / "checkpoint.pt", map_location=device, weights_only=False)
        model = build_transformer(
            input_dim=len(final_prepared.channel_names),
            num_classes=len(final_prepared.label_names),
            model_dim=final_candidate["model_dim"],
            num_heads=final_candidate["num_heads"],
            num_layers=final_candidate["num_layers"],
            dropout=final_candidate["dropout"],
            device=device,
        )
        model.load_state_dict(final_checkpoint["model_state_dict"])
        final_test_metrics = evaluate(model, test_loader, device)
        final_summary["final_selection"] = final_run
        final_summary["final_test"] = final_test_metrics
        (output_dir / "final_test_metrics.json").write_text(json.dumps(final_test_metrics, indent=2) + "\n")
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        del model
        gc.collect()

    (output_dir / "phase2_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps({"event": "phase2_complete", "summary": final_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
