"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
