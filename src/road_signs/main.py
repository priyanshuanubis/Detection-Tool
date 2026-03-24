"""CLI entrypoint for road-sign benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cnn import train_and_eval_cnn
from .config import load_config
from .data import read_split
from .evaluation import save_text_report
from .modern_models import train_yolo_classification, train_yolo_detection
from .traditional_cv import train_and_eval_traditional



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Road sign benchmark runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["traditional", "cnn", "yolo-detect", "yolo-cls", "all"],
        help="Which model family to run",
    )
    parser.add_argument(
        "--yolo-data",
        default=None,
        help="YOLO data yaml path for detection (required for yolo-detect/all)",
    )
    parser.add_argument(
        "--yolo-cls-root",
        default=None,
        help="YOLO classification dataset root (required for yolo-cls/all)",
    )
    return parser



def run_traditional(cfg):
    train_samples = read_split(cfg.paths.train_csv, cfg.paths.images_dir)
    val_samples = read_split(cfg.paths.val_csv, cfg.paths.images_dir)
    report = train_and_eval_traditional(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=cfg.paths.output_dir,
        image_size=cfg.training.image_size,
    )
    report_path = save_text_report(report, Path(cfg.paths.output_dir) / "traditional_report.txt")
    print(f"[Traditional] report saved to: {report_path}")



def run_cnn(cfg):
    train_samples = read_split(cfg.paths.train_csv, cfg.paths.images_dir)
    val_samples = read_split(cfg.paths.val_csv, cfg.paths.images_dir)

    result = train_and_eval_cnn(
        train_samples=train_samples,
        val_samples=val_samples,
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        lr=cfg.training.learning_rate,
        output_dir=cfg.paths.output_dir,
        seed=cfg.training.random_seed,
    )
    report_path = save_text_report(result.report, Path(cfg.paths.output_dir) / "cnn_report.txt")
    print(f"[CNN] best_val_acc={result.best_val_acc:.4f} model={result.model_path}")
    print(f"[CNN] report saved to: {report_path}")



def run_yolo_detect(cfg, yolo_data):
    if not yolo_data:
        raise ValueError("--yolo-data is required for yolo-detect mode")
    result = train_yolo_detection(
        data_yaml=yolo_data,
        output_dir=cfg.paths.output_dir,
        epochs=cfg.training.epochs,
    )
    print(f"[YOLO Detection] result: {result.save_dir}")



def run_yolo_cls(cfg, yolo_cls_root):
    if not yolo_cls_root:
        raise ValueError("--yolo-cls-root is required for yolo-cls mode")
    result = train_yolo_classification(
        cls_data_root=yolo_cls_root,
        output_dir=cfg.paths.output_dir,
        epochs=cfg.training.epochs,
    )
    print(f"[YOLO Classification] result: {result.save_dir}")



def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode in {"traditional", "all"}:
        run_traditional(cfg)

    if args.mode in {"cnn", "all"}:
        run_cnn(cfg)

    if args.mode in {"yolo-detect", "all"}:
        run_yolo_detect(cfg, args.yolo_data)

    if args.mode in {"yolo-cls", "all"}:
        run_yolo_cls(cfg, args.yolo_cls_root)


if __name__ == "__main__":
    main()
