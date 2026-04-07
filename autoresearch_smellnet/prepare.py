from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.dataset import PairedDataset, UniqueGCMSampler


RAW_SENSOR_COLUMNS = [
    "NO2",
    "C2H5OH",
    "VOC",
    "CO",
    "Alcohol",
    "LPG",
    "Benzene",
    "Temperature",
    "Pressure",
    "Humidity",
    "Gas_Resistance",
    "Altitude",
]

PAPER_SENSOR_COLUMNS = [
    "NO2",
    "C2H5OH",
    "VOC",
    "CO",
    "Alcohol",
    "LPG",
]

DROPPED_SENSOR_COLUMNS = [
    "Benzene",
    "Temperature",
    "Pressure",
    "Humidity",
    "Gas_Resistance",
    "Altitude",
]

DEFAULT_GRADIENT_PERIOD = 25
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 50


@dataclass(frozen=True)
class BaselineConfig:
    gradient_period: int = DEFAULT_GRADIENT_PERIOD
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    batch_size: int = 32


@dataclass(frozen=True)
class PreparedSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    channel_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    config: BaselineConfig


@dataclass(frozen=True)
class PreparedSearchSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    channel_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    config: BaselineConfig
    split_summary: dict


@dataclass(frozen=True)
class PreparedGCMSBank:
    X_gcms: np.ndarray
    label_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]


def _canonicalize_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"C2H50H": "C2H5OH"})
    kept = renamed.drop(columns=["timestamp", "State"], errors="ignore")
    missing = [col for col in RAW_SENSOR_COLUMNS if col not in kept.columns]
    if missing:
        raise ValueError(f"Missing expected sensor columns: {missing}")
    return kept[RAW_SENSOR_COLUMNS].copy()


def _subtract_first_row(df: pd.DataFrame) -> pd.DataFrame:
    return df - df.iloc[0]


def _apply_gradient(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    if periods <= 0:
        return df
    return df.diff(periods=periods).iloc[periods:].reset_index(drop=True)


def _prepare_sensor_frame(df: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    frame = _canonicalize_sensor_frame(df)
    frame = _subtract_first_row(frame)
    frame = frame.drop(columns=DROPPED_SENSOR_COLUMNS)
    frame = _apply_gradient(frame, config.gradient_period)
    return frame.reset_index(drop=True)


def _iter_label_csvs(root: Path, label_names: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for label in label_names:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        for csv_path in sorted(label_dir.glob("*.csv")):
            yield label, csv_path


def _group_csvs_by_label(root: Path, label_names: Iterable[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for label in label_names:
        label_dir = root / label
        grouped[label] = sorted(label_dir.glob("*.csv")) if label_dir.is_dir() else []
    return grouped


def _window_frame(df: pd.DataFrame, window_size: int, stride: int) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    if len(df) < window_size:
        return windows
    for start in range(0, len(df) - window_size + 1, stride):
        windows.append(df.iloc[start : start + window_size].values.astype(np.float32))
    return windows


def _build_split(root: Path, label_names: list[str], label_encoder: LabelEncoder, config: BaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    for label, csv_path in _iter_label_csvs(root, label_names):
        frame = pd.read_csv(csv_path)
        prepared = _prepare_sensor_frame(frame, config)
        windows = _window_frame(prepared, config.window_size, config.stride)
        X.extend(windows)
        y.extend([label] * len(windows))
    if not X:
        raise ValueError(f"No training windows built from {root}")
    return np.stack(X, axis=0), label_encoder.transform(y)


def _build_split_from_paths(
    csv_paths: list[tuple[str, Path]],
    label_encoder: LabelEncoder,
    config: BaselineConfig,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    for label, csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        prepared = _prepare_sensor_frame(frame, config)
        windows = _window_frame(prepared, config.window_size, config.stride)
        X.extend(windows)
        y.extend([label] * len(windows))
    if not X:
        raise ValueError(f"No windows built for {split_name}")
    return np.stack(X, axis=0), label_encoder.transform(y)


def _fit_train_scaler(X_train: np.ndarray) -> StandardScaler:
    n, t, c = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n * t, c))
    return scaler


def _apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, c = X.shape
    flat = scaler.transform(X.reshape(n * t, c))
    return flat.reshape(n, t, c).astype(np.float32)


def _collect_label_names(train_dir: Path) -> list[str]:
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def load_gcms_bank(gcms_csv: str | Path, *, expected_labels: list[str]) -> PreparedGCMSBank:
    gcms_path = Path(gcms_csv)
    df = pd.read_csv(gcms_path)
    label_col = df.columns[0]
    feature_cols = list(df.columns[1:])

    labels = df[label_col].astype(str).tolist()
    if sorted(labels) != sorted(expected_labels):
        raise ValueError(
            "GC-MS labels do not match the sensor labels. "
            f"Expected {len(expected_labels)} labels, got {len(labels)} labels from {gcms_path}."
        )

    label_to_row = {label: idx for idx, label in enumerate(labels)}
    ordered = df.set_index(label_col).loc[expected_labels, feature_cols].astype(np.float32)
    scaler = StandardScaler()
    X_gcms = scaler.fit_transform(ordered.values.astype(np.float32)).astype(np.float32)

    return PreparedGCMSBank(
        X_gcms=X_gcms,
        label_names=list(expected_labels),
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
    )


def _grouped_validation_paths(
    train_root: Path,
    label_names: list[str],
    *,
    validation_files_per_class: int,
    validation_fold_offset: int = 0,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], dict]:
    grouped = _group_csvs_by_label(train_root, label_names)
    train_paths: list[tuple[str, Path]] = []
    val_paths: list[tuple[str, Path]] = []
    split_summary: dict[str, dict] = {
        "validation_files_per_class": validation_files_per_class,
        "validation_fold_offset": validation_fold_offset,
        "labels": {},
    }

    for label in label_names:
        csvs = grouped[label]
        if len(csvs) <= validation_files_per_class:
            raise ValueError(
                f"Label '{label}' has only {len(csvs)} files, cannot reserve {validation_files_per_class} validation files."
            )

        n_csvs = len(csvs)
        fold_count = n_csvs
        start = validation_fold_offset % fold_count
        val_indices = [(start + i) % fold_count for i in range(validation_files_per_class)]
        val_index_set = set(val_indices)
        val_csvs = [csvs[idx] for idx in val_indices]
        train_csvs = [csv for idx, csv in enumerate(csvs) if idx not in val_index_set]
        train_paths.extend((label, path) for path in train_csvs)
        val_paths.extend((label, path) for path in val_csvs)
        split_summary["labels"][label] = {
            "train_files": [path.name for path in train_csvs],
            "val_files": [path.name for path in val_csvs],
            "fold_count": fold_count,
        }

    split_summary["train_file_count"] = len(train_paths)
    split_summary["val_file_count"] = len(val_paths)
    return train_paths, val_paths, split_summary


def prepare_baseline_splits(train_dir: str | Path, test_dir: str | Path, config: BaselineConfig | None = None) -> PreparedSplit:
    cfg = config or BaselineConfig()
    train_root = Path(train_dir)
    test_root = Path(test_dir)
    label_names = _collect_label_names(train_root)
    label_encoder = LabelEncoder()
    label_encoder.fit(label_names)

    X_train_raw, y_train = _build_split(train_root, label_names, label_encoder, cfg)
    X_test_raw, y_test = _build_split(test_root, label_names, label_encoder, cfg)

    scaler = _fit_train_scaler(X_train_raw)
    X_train = _apply_scaler(X_train_raw, scaler)
    X_test = _apply_scaler(X_test_raw, scaler)

    return PreparedSplit(
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        label_names=list(label_encoder.classes_),
        channel_names=list(PAPER_SENSOR_COLUMNS),
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        config=cfg,
    )


def prepare_search_splits(
    train_dir: str | Path,
    test_dir: str | Path,
    config: BaselineConfig | None = None,
    *,
    validation_files_per_class: int = 1,
    validation_fold_offset: int = 0,
) -> PreparedSearchSplit:
    cfg = config or BaselineConfig()
    train_root = Path(train_dir)
    test_root = Path(test_dir)
    label_names = _collect_label_names(train_root)
    label_encoder = LabelEncoder()
    label_encoder.fit(label_names)

    train_paths, val_paths, split_summary = _grouped_validation_paths(
        train_root,
        label_names,
        validation_files_per_class=validation_files_per_class,
        validation_fold_offset=validation_fold_offset,
    )
    test_paths = list(_iter_label_csvs(test_root, label_names))

    X_train_raw, y_train = _build_split_from_paths(train_paths, label_encoder, cfg, split_name="train")
    X_val_raw, y_val = _build_split_from_paths(val_paths, label_encoder, cfg, split_name="validation")
    X_test_raw, y_test = _build_split_from_paths(test_paths, label_encoder, cfg, split_name="test")

    scaler = _fit_train_scaler(X_train_raw)
    X_train = _apply_scaler(X_train_raw, scaler)
    X_val = _apply_scaler(X_val_raw, scaler)
    X_test = _apply_scaler(X_test_raw, scaler)

    split_summary["train_windows"] = int(X_train.shape[0])
    split_summary["val_windows"] = int(X_val.shape[0])
    split_summary["test_windows"] = int(X_test.shape[0])

    return PreparedSearchSplit(
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        label_names=list(label_encoder.classes_),
        channel_names=list(PAPER_SENSOR_COLUMNS),
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        config=cfg,
        split_summary=split_summary,
    )


def make_dataloaders(prepared: PreparedSplit) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(prepared.X_train),
        torch.from_numpy(prepared.y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(prepared.X_test),
        torch.from_numpy(prepared.y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=prepared.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=prepared.config.batch_size, shuffle=False)
    return train_loader, test_loader


def make_search_dataloaders(prepared: PreparedSearchSplit) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(prepared.X_train),
        torch.from_numpy(prepared.y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(prepared.X_val),
        torch.from_numpy(prepared.y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(prepared.X_test),
        torch.from_numpy(prepared.y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=prepared.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=prepared.config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=prepared.config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def make_contrastive_train_loader(
    prepared: PreparedSearchSplit,
    gcms_bank: PreparedGCMSBank,
) -> DataLoader:
    if prepared.label_names != gcms_bank.label_names:
        raise ValueError("Sensor and GC-MS label order mismatch.")

    pair_data = [
        (gcms_bank.X_gcms[int(label_idx)], sensor_window)
        for sensor_window, label_idx in zip(prepared.X_train, prepared.y_train, strict=True)
    ]
    train_dataset = PairedDataset(pair_data)
    sampler = UniqueGCMSampler(train_dataset.data, batch_size=prepared.config.batch_size)
    return DataLoader(train_dataset, batch_size=prepared.config.batch_size, sampler=sampler)


def accuracy_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item() * 100.0)


def save_run_metadata(
    output_dir: str | Path,
    *,
    prepared: PreparedSplit,
    model_config: dict,
    metrics: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels_payload = {"labels": prepared.label_names}
    preprocessing_payload = {
        "track": "exact-upstream",
        "raw_sensor_columns": RAW_SENSOR_COLUMNS,
        "used_sensor_columns": prepared.channel_names,
        "dropped_sensor_columns": DROPPED_SENSOR_COLUMNS,
        "gradient_period": prepared.config.gradient_period,
        "window_size": prepared.config.window_size,
        "stride": prepared.config.stride,
        "scaler_mean": prepared.scaler_mean,
        "scaler_scale": prepared.scaler_scale,
        "labels": prepared.label_names,
        "model": model_config,
    }

    (out / "labels.json").write_text(json.dumps(labels_payload, indent=2) + "\n")
    (out / "preprocessing.json").write_text(json.dumps(preprocessing_payload, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
