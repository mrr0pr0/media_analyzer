from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from utils import MediaAnalyzerError


def extract_frames(
    video_path: Path,
    interval: int = 1,
    *,
    show_progress: bool = True,
) -> tuple[list[Path], Path | None]:
    """
    Extract frames using FFmpeg into a temp directory.

    Returns (frame_paths, temp_dir).
    """
    video_path = Path(video_path)
    if interval <= 0:
        raise MediaAnalyzerError("--interval must be a positive integer (seconds).")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MediaAnalyzerError(
            "FFmpeg is required but was not found on PATH. Install FFmpeg and try again."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="media_analyzer_frames_"))
    out_pattern = tmp_dir / "frame_%06d.jpg"

    # Use fps filter to grab one frame every N seconds: fps=1/interval
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        str(out_pattern),
    ]

    if show_progress:
        print(f"Extracting frames (every {interval}s)...")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        _safe_rmtree(tmp_dir)
        raise MediaAnalyzerError(f"Failed to run FFmpeg: {e}") from e

    if proc.returncode != 0:
        _safe_rmtree(tmp_dir)
        msg = (proc.stderr or proc.stdout or "").strip()
        raise MediaAnalyzerError(f"FFmpeg failed extracting frames.\n{msg}")

    frames = sorted(tmp_dir.glob("frame_*.jpg"))

    if show_progress:
        print(f"Extracted {len(frames)} frame(s).")

    return frames, tmp_dir


def cleanup_frames_dir(tmp_dir: Path | None) -> None:
    if tmp_dir is None:
        return
    _safe_rmtree(Path(tmp_dir))


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
