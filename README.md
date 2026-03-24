# Road Sign Detection and Classification Benchmark

This project provides full code for comparing road-sign recognition approaches:

1. Traditional Computer Vision + ML (HOG + color histogram + edge features + SVM)
2. CNN classifier (PyTorch)
3. Modern Vision Models
   - YOLOv8 detection
   - YOLOv8 classification
   - SAM (Segment Anything) mask extraction over YOLO detections

It supports your 81-class road-sign label set and now includes utilities for the dataset layout you described (`archive/Indian Road Signs/Images`).

---

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## 2) Expected archive test dataset layout

When you add data to the repository, place it like this:

```text
archive/
├── labels.csv (or labels.xlsx)
└── Indian Road Signs/
    └── Images/
        ├── 1.png
        ├── 2.png
        ├── ...
        └── 81.png
```

Label sheet must contain either:
- `id,label`
- or `label_id,label`

Image filenames can be numeric (`68.png`) or contain a single number (`sign_68.png`).

---

## 3) Config

Edit `configs/experiment.yaml`:

```yaml
paths:
  dataset_root: ../data
  archie_root: ../archive
  train_csv: ../data/train.csv
  val_csv: ../data/val.csv
  test_csv: ../data/test.csv
  images_dir: ../data/images
  output_dir: ../runs
```

> If you are using archie-generated CSVs, set `images_dir` to the `archive` root (because generated image paths are relative to that root).

---

## 4) Prepare CSV splits from archive folder

Generate `train.csv`, `val.csv`, `test.csv` directly from your `archive` dataset:

```bash
python -m road_signs.main \
  --config configs/experiment.yaml \
  --mode prepare-archie \
  --archie-root archive \
  --csv-out-dir data
```

This writes:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/archie_dataset_info.txt`

Because this dataset often has one image per class, the code duplicates all samples across train/val/test (shuffled order) so training/testing pipelines can run end-to-end.

Optional: build YOLO-classification folder tree from generated CSVs:

```bash
python -m road_signs.main \
  --config configs/experiment.yaml \
  --mode prepare-archie \
  --archie-root archive \
  --csv-out-dir data \
  --build-yolo-cls-tree
```

---

## 5) Run experiments

### Traditional CV baseline

```bash
python -m road_signs.main --config configs/experiment.yaml --mode traditional
```

Outputs:
- `runs/traditional_svm.joblib`
- `runs/traditional_report.txt`

### CNN baseline

```bash
python -m road_signs.main --config configs/experiment.yaml --mode cnn
```

Outputs:
- `runs/cnn_best.pt`
- `runs/cnn_report.txt`

### YOLOv8 detection

Prepare YOLO `data.yaml`, then:

```bash
python -m road_signs.main \
  --config configs/experiment.yaml \
  --mode yolo-detect \
  --yolo-data data_yolo.yaml
```

### YOLOv8 classification

YOLO classification expects:

```text
<root>/
  train/<class_name>/*.jpg|png
  val/<class_name>/*.jpg|png
```

Run:

```bash
python -m road_signs.main \
  --config configs/experiment.yaml \
  --mode yolo-cls \
  --yolo-cls-root path/to/cls_root
```

### Convenience runner

```bash
python scripts/run_experiments.py --config configs/experiment.yaml --prepare-archie --archie-root archive
```

---

## 6) SAM + YOLO integration example

```python
from road_signs.modern_models import sam_mask_from_yolo_boxes

results = sam_mask_from_yolo_boxes(
    image_path="archive/Indian Road Signs/Images/1.png",
    yolo_model_path="runs/yolo_detection/weights/best.pt",
    sam_model_path="sam_vit_b_01ec64.pth",
    device="cuda",  # or "cpu"
)

for r in results:
    print(r["class_name"], r["confidence"], r["bbox_xyxy"])
```

---

## 7) Notes

- Traditional/CNN modules output classification reports.
- YOLO training uses Ultralytics built-in metrics.
- SAM requires a downloaded checkpoint.
- This code is now amended to directly support your incoming `archive/Indian Road Signs/Images` folder format.
