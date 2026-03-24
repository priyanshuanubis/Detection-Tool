"""Convenience runner for all road sign experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path



def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--yolo-data", default=None)
    parser.add_argument("--yolo-cls-root", default=None)
    args = parser.parse_args()

    base = [sys.executable, "-m", "road_signs.main", "--config", args.config]

    run(base + ["--mode", "traditional"])
    run(base + ["--mode", "cnn"])

    if args.yolo_data:
        run(base + ["--mode", "yolo-detect", "--yolo-data", args.yolo_data])

    if args.yolo_cls_root:
        run(base + ["--mode", "yolo-cls", "--yolo-cls-root", args.yolo_cls_root])


if __name__ == "__main__":
    main()
