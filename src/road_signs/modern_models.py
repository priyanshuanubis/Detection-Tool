"""Modern model workflows: YOLOv8 detection/classification and SAM-assisted mask extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .data import Sample, load_image_bgr
from .label_map import LABELS



def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def export_yolo_detection_labels(samples: list[Sample], output_dir: str | Path) -> Path:
    """Export samples into YOLO detection text labels.

    Requires bounding box fields in samples. Labels are 0-indexed for YOLO.
    """
    out = Path(output_dir)
    images_out = out / "images"
    labels_out = out / "labels"
    _ensure_dir(images_out)
    _ensure_dir(labels_out)

    for s in samples:
        if None in (s.xmin, s.ymin, s.xmax, s.ymax):
            raise ValueError("YOLO detection export requires bbox columns in CSV.")

        image = load_image_bgr(s.image_path)
        h, w = image.shape[:2]

        img_out = images_out / s.image_path.name
        cv2.imwrite(str(img_out), image)

        x1, y1, x2, y2 = s.xmin, s.ymin, s.xmax, s.ymax
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        txt_out = labels_out / (s.image_path.stem + ".txt")
        with txt_out.open("w", encoding="utf-8") as fh:
            fh.write(f"{s.label_id - 1} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return out



def create_yolo_data_yaml(train_dir: Path, val_dir: Path, out_file: str | Path) -> Path:
    out_file = Path(out_file)
    data = {
        "path": str(out_file.parent.resolve()),
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "names": {idx - 1: name for idx, name in LABELS.items()},
    }
    with out_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out_file



def train_yolo_detection(
    data_yaml: str | Path,
    output_dir: str | Path,
    model: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
):
    yolo = YOLO(model)
    return yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir),
        name="yolo_detection",
        exist_ok=True,
    )



def train_yolo_classification(
    cls_data_root: str | Path,
    output_dir: str | Path,
    model: str = "yolov8n-cls.pt",
    epochs: int = 20,
    imgsz: int = 224,
):
    """Train YOLOv8 classifier.

    cls_data_root must follow this structure:
    root/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
    """
    yolo = YOLO(model)
    return yolo.train(
        data=str(cls_data_root),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir),
        name="yolo_classification",
        exist_ok=True,
    )



def sam_mask_from_yolo_boxes(
    image_path: str | Path,
    yolo_model_path: str | Path,
    sam_model_path: str | Path,
    device: str = "cpu",
):
    """Run YOLO detection and create SAM masks for detected signs.

    Returns list of dicts containing class id, confidence, bbox and mask.
    """
    from segment_anything import SamPredictor, sam_model_registry  # local import for optional dependency

    image_bgr = load_image_bgr(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    det_model = YOLO(str(yolo_model_path))
    det_results = det_model.predict(source=image_rgb, verbose=False)
    boxes = det_results[0].boxes

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_model_path))
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    outputs = []
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy().reshape(-1)
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        masks, scores, _ = predictor.predict(
            box=xyxy,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        outputs.append(
            {
                "class_id": cls_id,
                "class_name": LABELS[cls_id + 1],
                "confidence": conf,
                "bbox_xyxy": xyxy.tolist(),
                "mask": masks[best_idx].astype(np.uint8),
            }
        )

    return outputs
