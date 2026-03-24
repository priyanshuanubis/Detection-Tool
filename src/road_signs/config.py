"""Configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    dataset_root: Path
    archie_root: Path | None
    train_csv: Path
    val_csv: Path
    test_csv: Path
    images_dir: Path
    output_dir: Path


@dataclass
class TrainingConfig:
    image_size: int = 64
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    training: TrainingConfig



def _abs(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (base / path).resolve()



def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base = config_path.parent
    paths = raw["paths"]
    archie_raw = paths.get("archie_root")

    path_cfg = PathsConfig(
        dataset_root=_abs(base, paths["dataset_root"]),
        archie_root=_abs(base, archie_raw) if archie_raw else None,
        train_csv=_abs(base, paths["train_csv"]),
        val_csv=_abs(base, paths["val_csv"]),
        test_csv=_abs(base, paths["test_csv"]),
        images_dir=_abs(base, paths["images_dir"]),
        output_dir=_abs(base, paths["output_dir"]),
    )

    train_raw = raw.get("training", {})
    train_cfg = TrainingConfig(
        image_size=int(train_raw.get("image_size", 64)),
        batch_size=int(train_raw.get("batch_size", 32)),
        epochs=int(train_raw.get("epochs", 15)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        random_seed=int(train_raw.get("random_seed", 42)),
    )

    return ExperimentConfig(paths=path_cfg, training=train_cfg)
