from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from autoresearch_smellnet.prepare import (
    BaselineConfig,
    PAPER_SENSOR_COLUMNS,
    RAW_SENSOR_COLUMNS,
    DROPPED_SENSOR_COLUMNS,
    make_search_dataloaders,
    prepare_search_splits,
)
from autoresearch_smellnet.train import (
    best_state_dict,
    build_transformer,
    evaluate,
    pick_device,
    set_seed,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budgeted SmellNet autoresearch loop.")
    parser.add_argument("--train-dir", default="data/offline_training")
    parser.add_argument("--test-dir", default="data/offline_testing")
    parser.add_argument("--output-dir", default="autoresearch_runs/search_exact_upstream")
    parser.add_argument("--time-budget-hours", type=float, default=1.0)
    parser.add_argument("--trial-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--validation-files-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def candidate_stream(rng: random.Random):
    yield {
        "model_dim": 256,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 3e-4,
    }

    spaces = {
        128: [4, 8],
        256: [4, 8],
        384: [6, 8],
        512: [8],
    }
    lrs = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    layers = [2, 3, 4, 5]

    while True:
        model_dim = rng.choice(sorted(spaces))
        yield {
            "model_dim": model_dim,
            "num_heads": rng.choice(spaces[model_dim]),
            "num_layers": rng.choice(layers),
            "dropout": rng.choice(dropouts),
            "lr": rng.choice(lrs),
        }


def _preprocessing_payload(prepared, model_config: dict) -> dict:
    return {
        "track": "exact-upstream-search",
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


def save_trial_artifacts(
    trial_dir: Path,
    *,
    prepared,
    model_config: dict,
    metrics: dict,
    checkpoint_payload: dict,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, trial_dir / "checkpoint.pt")
    (trial_dir / "labels.json").write_text(json.dumps({"labels": prepared.label_names}, indent=2) + "\n")
    (trial_dir / "preprocessing.json").write_text(json.dumps(_preprocessing_payload(prepared, model_config), indent=2) + "\n")
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = BaselineConfig(
        gradient_period=args.gradient_period,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    prepared = prepare_search_splits(
        args.train_dir,
        args.test_dir,
        baseline_config,
        validation_files_per_class=args.validation_files_per_class,
    )
    train_loader, val_loader, test_loader = make_search_dataloaders(prepared)

    budget_seconds = max(args.time_budget_hours, 0.0) * 3600.0
    started_at = time.time()
    rng = random.Random(args.seed)

    best_trial: dict | None = None
    trials: list[dict] = []

    print(
        json.dumps(
            {
                "event": "search_start",
                "device": str(device),
                "budget_hours": args.time_budget_hours,
                "trial_epochs": args.trial_epochs,
                "validation_files_per_class": args.validation_files_per_class,
                "split_summary": prepared.split_summary,
            }
        ),
        flush=True,
    )

    for trial_index, candidate in enumerate(candidate_stream(rng), start=1):
        elapsed = time.time() - started_at
        if budget_seconds > 0 and elapsed >= budget_seconds:
            break
        if args.max_trials is not None and trial_index > args.max_trials:
            break

        trial_seed = args.seed + trial_index - 1
        set_seed(trial_seed)
        trial_dir = output_dir / f"trial_{trial_index:03d}"

        model = build_transformer(
            input_dim=len(prepared.channel_names),
            num_classes=len(prepared.label_names),
            model_dim=candidate["model_dim"],
            num_heads=candidate["num_heads"],
            num_layers=candidate["num_layers"],
            dropout=candidate["dropout"],
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=candidate["lr"])

        history: list[dict] = []
        best_val_metrics: dict | None = None
        best_epoch = 0
        best_weights = None

        print(
            json.dumps(
                {
                    "event": "trial_start",
                    "trial": trial_index,
                    "seed": trial_seed,
                    "candidate": candidate,
                    "elapsed_seconds": round(elapsed, 2),
                }
            ),
            flush=True,
        )

        for epoch in range(1, args.trial_epochs + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc@1": train_metrics["acc@1"],
                "val_loss": val_metrics["loss"],
                "val_acc@1": val_metrics["acc@1"],
                "val_acc@5": val_metrics["acc@5"],
            }
            history.append(epoch_metrics)
            print(
                json.dumps(
                    {
                        "event": "trial_epoch",
                        "trial": trial_index,
                        "epoch": epoch,
                        "train_acc@1": round(train_metrics["acc@1"], 4),
                        "val_acc@1": round(val_metrics["acc@1"], 4),
                        "val_acc@5": round(val_metrics["acc@5"], 4),
                    }
                ),
                flush=True,
            )

            if best_val_metrics is None or (
                val_metrics["acc@1"],
                val_metrics["acc@5"],
            ) > (
                best_val_metrics["acc@1"],
                best_val_metrics["acc@5"],
            ):
                best_val_metrics = val_metrics
                best_epoch = epoch
                best_weights = best_state_dict(model)

        assert best_val_metrics is not None
        assert best_weights is not None
        model_config = {
            "track": "exact-upstream-search",
            "contrastive": False,
            "architecture": "transformer",
            "input_dim": len(prepared.channel_names),
            "model_dim": candidate["model_dim"],
            "num_heads": candidate["num_heads"],
            "num_layers": candidate["num_layers"],
            "dropout": candidate["dropout"],
            "num_classes": len(prepared.label_names),
        }
        checkpoint_payload = {
            "model_state_dict": best_weights,
            "model_config": model_config,
            "labels": prepared.label_names,
        }

        trial_record = {
            "trial": trial_index,
            "seed": trial_seed,
            "candidate": candidate,
            "best_epoch": best_epoch,
            "validation": best_val_metrics,
            "history": history,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        trials.append(trial_record)
        save_trial_artifacts(
            trial_dir,
            prepared=prepared,
            model_config=model_config,
            metrics=trial_record,
            checkpoint_payload=checkpoint_payload,
        )

        if best_trial is None or (
            best_val_metrics["acc@1"],
            best_val_metrics["acc@5"],
        ) > (
            best_trial["validation"]["acc@1"],
            best_trial["validation"]["acc@5"],
        ):
            best_trial = trial_record
            torch.save(checkpoint_payload, output_dir / "best_checkpoint.pt")
            (output_dir / "best_trial.json").write_text(json.dumps(best_trial, indent=2) + "\n")

        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        del model
        gc.collect()

    summary = {
        "track": "exact-upstream-search",
        "device": str(device),
        "time_budget_hours": args.time_budget_hours,
        "trial_epochs": args.trial_epochs,
        "validation_files_per_class": args.validation_files_per_class,
        "trial_count": len(trials),
        "best_trial": best_trial,
        "trials": trials,
    }

    if best_trial is not None:
        best_model = build_transformer(
            input_dim=len(prepared.channel_names),
            num_classes=len(prepared.label_names),
            model_dim=best_trial["candidate"]["model_dim"],
            num_heads=best_trial["candidate"]["num_heads"],
            num_layers=best_trial["candidate"]["num_layers"],
            dropout=best_trial["candidate"]["dropout"],
            device=device,
        )
        best_checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
        best_model.load_state_dict(best_checkpoint["model_state_dict"])
        final_test_metrics = evaluate(best_model, test_loader, device)
        summary["final_test"] = final_test_metrics
        (output_dir / "final_test_metrics.json").write_text(json.dumps(final_test_metrics, indent=2) + "\n")
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        del best_model
        gc.collect()

    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        json.dumps(
            {
                "event": "search_complete",
                "trial_count": len(trials),
                "best_trial": None if best_trial is None else {
                    "trial": best_trial["trial"],
                    "best_epoch": best_trial["best_epoch"],
                    "validation": best_trial["validation"],
                    "candidate": best_trial["candidate"],
                },
                "final_test": summary.get("final_test"),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
