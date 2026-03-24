"""Traditional computer vision baseline (HOG + color histogram + SVM)."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from .data import Sample, crop_bbox, load_image_bgr
from .label_map import LABELS


class TraditionalCVClassifier:
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self.hog = cv2.HOGDescriptor(
            _winSize=(image_size, image_size),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("svm", LinearSVC(C=1.0, random_state=42, max_iter=3000)),
            ]
        )

    def _extract_single(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (self.image_size, self.image_size))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog_features = self.hog.compute(gray).flatten()

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        edges = cv2.Canny(gray, 100, 200).astype(np.float32).flatten() / 255.0

        return np.concatenate([hog_features, hist, edges])

    def build_matrix(self, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        labels: list[int] = []

        for s in tqdm(samples, desc="Extracting traditional features"):
            image = load_image_bgr(s.image_path)
            image = crop_bbox(image, s)
            feats.append(self._extract_single(image))
            labels.append(s.label_id - 1)

        return np.stack(feats), np.array(labels)

    def fit(self, train_samples: list[Sample]) -> None:
        x_train, y_train = self.build_matrix(train_samples)
        self.model.fit(x_train, y_train)

    def evaluate(self, eval_samples: list[Sample]) -> str:
        x_eval, y_eval = self.build_matrix(eval_samples)
        pred = self.model.predict(x_eval)
        target_names = [LABELS[i + 1] for i in sorted(set(y_eval))]
        return classification_report(
            y_eval,
            pred,
            labels=sorted(set(y_eval)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )

    def save(self, out_file: str | Path) -> None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"image_size": self.image_size, "model": self.model}, out_file)


def train_and_eval_traditional(
    train_samples: list[Sample],
    val_samples: list[Sample],
    output_dir: str | Path,
    image_size: int = 64,
) -> str:
    clf = TraditionalCVClassifier(image_size=image_size)
    clf.fit(train_samples)
    report = clf.evaluate(val_samples)
    clf.save(Path(output_dir) / "traditional_svm.joblib")
    return report
