"""Evaluation helpers and report persistence."""

from __future__ import annotations

from pathlib import Path



def save_text_report(report: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path
