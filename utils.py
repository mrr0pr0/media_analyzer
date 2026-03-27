from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEOS = {".mp4", ".mov", ".mkv"}


class MediaAnalyzerError(Exception):
    pass


def print_banner() -> None:
    print("Media Analyzer")
    print("--------------")


def validate_file(path: Path) -> None:
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise MediaAnalyzerError(f"File not found: {path}")

    if not path.is_file():
        raise MediaAnalyzerError(f"Not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGES and ext not in SUPPORTED_VIDEOS:
        supported = ", ".join(sorted(SUPPORTED_IMAGES | SUPPORTED_VIDEOS))
        raise MediaAnalyzerError(
            f"Unsupported file format: {ext or '<no extension>'}. Supported: {supported}"
        )


def get_file_type(path: Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_IMAGES:
        return "image"
    if ext in SUPPORTED_VIDEOS:
        return "video"
    raise MediaAnalyzerError(f"Unsupported file format: {ext or '<no extension>'}")


def format_results(results: Iterable[dict]) -> str:
    lines: list[str] = []
    for r in results:
        label = str(r.get("label", "")).strip()
        conf = float(r.get("confidence", 0.0))
        lines.append(f"  {label:<12} ({conf:.2f})")
    return "\n".join(lines)


def save_json(payload: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
