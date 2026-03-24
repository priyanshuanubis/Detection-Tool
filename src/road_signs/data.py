"""Dataset loading, preprocessing, and archie-format preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import LABELS


@dataclass
class Sample:
    image_path: Path
    label_id: int
    xmin: int | None = None
    ymin: int | None = None
    xmax: int | None = None
    ymax: int | None = None


REQUIRED_COLUMNS = {"image", "label_id"}
OPTIONAL_BBOX_COLUMNS = {"xmin", "ymin", "xmax", "ymax"}



def read_split(csv_path: str | Path, images_dir: str | Path) -> list[Sample]:
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    samples: list[Sample] = []
    has_bbox = OPTIONAL_BBOX_COLUMNS.issubset(df.columns)

    for row in df.itertuples(index=False):
        img = images_dir / getattr(row, "image")
        if has_bbox:
            samples.append(
                Sample(
                    image_path=img,
                    label_id=int(getattr(row, "label_id")),
                    xmin=int(getattr(row, "xmin")),
                    ymin=int(getattr(row, "ymin")),
                    xmax=int(getattr(row, "xmax")),
                    ymax=int(getattr(row, "ymax")),
                )
            )
        else:
            samples.append(
                Sample(image_path=img, label_id=int(getattr(row, "label_id")))
            )

    return samples



def load_image_bgr(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return arr



def crop_bbox(image: np.ndarray, sample: Sample) -> np.ndarray:
    if None in (sample.xmin, sample.ymin, sample.xmax, sample.ymax):
        return image

    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, sample.xmin or 0))
    y1 = max(0, min(h - 1, sample.ymin or 0))
    x2 = max(x1 + 1, min(w, sample.xmax or w))
    y2 = max(y1 + 1, min(h, sample.ymax or h))
    return image[y1:y2, x1:x2]


class SignClassificationDataset(Dataset):
    """Torch dataset for road sign classification."""

    def __init__(self, samples: list[Sample], image_size: int, transforms=None):
        self.samples = samples
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = load_image_bgr(s.image_path)
        image = crop_bbox(image, s)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)

        if self.transforms is not None:
            tensor = self.transforms(pil)
        else:
            arr = np.array(pil.resize((self.image_size, self.image_size))) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        return tensor, int(s.label_id) - 1



def _find_labels_file(archie_root: Path) -> Path:
    candidates = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        candidates.extend(archie_root.glob(ext))
        candidates.extend(archie_root.glob(f"**/{ext}"))

    preferred = [p for p in candidates if "label" in p.stem.lower()]
    if preferred:
        return sorted(preferred, key=lambda p: len(p.parts))[0]
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0]
    raise FileNotFoundError(f"No labels sheet found under {archie_root}")



def _load_labels_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    normalized = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=normalized)

    if "id" in df.columns and "label" in df.columns:
        out = df[["id", "label"]].copy()
        out["id"] = out["id"].astype(int)
        return out

    if "label_id" in df.columns and "label" in df.columns:
        out = df[["label_id", "label"]].rename(columns={"label_id": "id"}).copy()
        out["id"] = out["id"].astype(int)
        return out

    raise ValueError(
        f"Unsupported label sheet columns in {path}. Expected columns like [id, label] or [label_id, label]."
    )



def _find_images_dir(archie_root: Path) -> Path:
    preferred = archie_root / "Indian Road Signs" / "Images"
    if preferred.exists():
        return preferred

    legacy = archie_root / "images" / "Indian Road Signs"
    if legacy.exists():
        return legacy

    for p in archie_root.glob("**/*"):
        if p.is_dir() and p.name.lower() in {"images", "indian road signs"}:
            # prefer full expected depth .../Indian Road Signs/Images
            if p.name.lower() == "images" and p.parent.name.lower() == "indian road signs":
                return p
    raise FileNotFoundError(
        f"Could not find image directory under {archie_root}. Expected '{archie_root}/Indian Road Signs/Images'."
    )



def _index_images(images_dir: Path) -> dict[int, Path]:
    mapped: dict[int, Path] = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem.strip()
        if stem.isdigit():
            mapped[int(stem)] = p
            continue

        digits = re.findall(r"\d+", stem)
        if len(digits) == 1:
            mapped[int(digits[0])] = p
    return mapped



def generate_archie_splits(
    archie_root: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[Path, Path, Path]:
    """Generate train/val/test CSVs for the archie folder layout.

    Expected incoming layout (user-provided):
      <archie_root>/labels.*
      <archie_root>/Indian Road Signs/Images/{1..81}.png

    Since this dataset commonly has one image per class, the function duplicates rows
    across train/val/test to allow pipeline smoke-testing and training code execution.
    """
    archie_root = Path(archie_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_file = _find_labels_file(archie_root)
    label_df = _load_labels_sheet(labels_file)

    images_dir = _find_images_dir(archie_root)
    image_map = _index_images(images_dir)

    rows = []
    for row in label_df.itertuples(index=False):
        label_id = int(row.id)
        if label_id not in LABELS:
            continue
        image_path = image_map.get(label_id)
        if image_path is None:
            continue
        rel = image_path.relative_to(archie_root)
        rows.append({"image": str(rel).replace("\\", "/"), "label_id": label_id})

    if not rows:
        raise ValueError("No matching image-label rows were found in the archie dataset.")

    all_df = pd.DataFrame(rows).sort_values("label_id").reset_index(drop=True)

    # with one image per class, stratified splits are impossible; duplicate for usability
    train_df = all_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = all_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    test_df = all_df.sample(frac=1.0, random_state=13).reset_index(drop=True)

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    meta = out_dir / "archie_dataset_info.txt"
    meta.write_text(
        "\n".join(
            [
                f"archie_root={archie_root.resolve()}",
                f"labels_file={labels_file.resolve()}",
                f"images_dir={images_dir.resolve()}",
                f"num_samples={len(all_df)}",
                "split_note=single-image-per-class detected; train/val/test each contain all samples in shuffled order",
                f"requested_ratios=train:{train_ratio},val:{val_ratio},test:{test_ratio}",
            ]
        ),
        encoding="utf-8",
    )

    return train_csv, val_csv, test_csv



def build_yolo_cls_tree_from_csvs(
    dataset_root: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    out_root: str | Path,
    copy_files: bool = True,
) -> Path:
    """Create YOLO classification folder tree from CSV splits.

    Output structure:
      out_root/train/<label_name>/*
      out_root/val/<label_name>/*
    """
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)

    for split_name, csv_path in (("train", train_csv), ("val", val_csv)):
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            image_rel = Path(getattr(row, "image"))
            src = dataset_root / image_rel
            label_id = int(getattr(row, "label_id"))
            class_dir = out_root / split_name / LABELS[label_id].replace("/", "-")
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / src.name
            if copy_files:
                if src.resolve() != dst.resolve():
                    dst.write_bytes(src.read_bytes())
            else:
                if dst.exists() or src.resolve() == dst.resolve():
                    continue
                dst.symlink_to(src.resolve())

    return out_root
