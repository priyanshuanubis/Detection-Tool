# Road Sign Detection and Classification Benchmark

This project provides **full code** for comparing road-sign recognition approaches:

1. **Traditional Computer Vision + ML** (HOG + color histogram + edge features + SVM)
2. **CNN classifier** (PyTorch)
3. **Modern Vision Models**
   - **YOLOv8 detection**
   - **YOLOv8 classification**
   - **SAM (Segment Anything) mask extraction over YOLO detections**

It is designed for your 81-class road-sign label set (included in `data/labels.csv`).

---

## 1) Project Structure

```text
Detection-Tool/
├── configs/
│   └── experiment.yaml
├── data/
│   └── labels.csv
├── scripts/
│   └── run_experiments.py
├── src/
│   └── road_signs/
│       ├── __init__.py
│       ├── cnn.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── label_map.py
│       ├── main.py
│       ├── modern_models.py
│       └── traditional_cv.py
└── pyproject.toml
```

---

## 2) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## 3) Dataset Format

Put your images in:

```text
data/images/
```

Create split CSV files:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

### Minimum CSV columns (classification)

```csv
image,label_id
img_0001.jpg,1
img_0002.jpg,24
```

### Optional bbox columns (for detection / better cropping)

```csv
image,label_id,xmin,ymin,xmax,ymax
img_0001.jpg,1,12,20,96,104
```

> `label_id` should be in **[1..81]**.

---

## 4) Configure Paths/Training

Edit `configs/experiment.yaml`:

```yaml
paths:
  dataset_root: ../data
  train_csv: ../data/train.csv
  val_csv: ../data/val.csv
  test_csv: ../data/test.csv
  images_dir: ../data/images
  output_dir: ../runs

training:
  image_size: 64
  batch_size: 32
  epochs: 15
  learning_rate: 0.001
  random_seed: 42
```

---

## 5) Run Experiments

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

Prepare a valid YOLO data yaml (e.g., `data_yolo.yaml`), then:

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
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```

Run:

```bash
python -m road_signs.main \
  --config configs/experiment.yaml \
  --mode yolo-cls \
  --yolo-cls-root path/to/cls_root
```

### Quick multi-run helper

```bash
python scripts/run_experiments.py --config configs/experiment.yaml
```

---

## 6) SAM + YOLO integration example

In Python:

```python
from road_signs.modern_models import sam_mask_from_yolo_boxes

results = sam_mask_from_yolo_boxes(
    image_path="data/images/sample.jpg",
    yolo_model_path="runs/yolo_detection/weights/best.pt",
    sam_model_path="sam_vit_b_01ec64.pth",
    device="cuda",  # or "cpu"
)

for r in results:
    print(r["class_name"], r["confidence"], r["bbox_xyxy"])
```

---

## 7) Notes on Evaluation

- Traditional/CNN modules currently generate a **classification report** (precision, recall, f1).
- YOLO training uses Ultralytics built-in metrics.
- You can extend this with:
  - confusion matrix plots,
  - mAP@50 and mAP@50-95 comparisons,
  - latency (CPU/GPU) benchmarking,
  - per-class error analysis.

---

## 8) Label Set

The complete 81 labels are provided in:
- `data/labels.csv`
- `src/road_signs/label_map.py`

---

## 9) Common Troubleshooting

- **Module not found `road_signs`**: install with `pip install -e .`
- **YOLO training errors**: verify dataset folder structure and YAML paths
- **SAM checkpoint missing**: download a SAM checkpoint and pass its path
- **CUDA not available**: code falls back to CPU for CNN (YOLO/SAM depend on runtime settings)

