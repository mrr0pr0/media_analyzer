from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import video
import vision
from utils import MediaAnalyzerError, format_results, get_file_type, print_banner, save_json, validate_file


def _dedupe_by_label_keep_max(results: list[dict]) -> list[dict]:
    best: dict[str, float] = {}
    for r in results:
        label = str(r.get("label", "")).strip()
        conf = float(r.get("confidence", 0.0))
        if not label:
            continue
        if label not in best or conf > best[label]:
            best[label] = conf
    out = [{"label": k, "confidence": float(v)} for k, v in best.items()]
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out


def _copy_saved_frames(
    *,
    src_dir: Path,
    dest_dir: Path,
    pattern: str,
    show_progress: bool,
) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob(pattern))
    for i, p in enumerate(files, start=1):
        shutil.copy2(p, dest_dir / p.name)
        if show_progress and len(files) >= 10 and i % 10 == 0:
            print(f"Saved {i}/{len(files)} frame(s)...")
    return len(files)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="media_analyzer", description="Analyze images/videos with YOLOv11.")
    p.add_argument("file", help="Path to an image or video file")
    p.add_argument("--interval", type=int, default=1, help="Frame extraction interval in seconds (default: 1)")
    p.add_argument("--output", type=str, default=None, help="Save results to a JSON file")
    p.add_argument("--save-frames", action="store_true", help="Save analyzed frames as images")
    p.add_argument("--draw-boxes", action="store_true", help="Draw bounding boxes on detected objects")
    p.add_argument("--no-progress", action="store_true", help="Disable progress indicator")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print_banner()

    try:
        path = Path(args.file).expanduser().resolve()
        validate_file(path)
        file_type = get_file_type(path)

        model = vision.load_model()

        frames_analyzed = 0
        objects: list[dict] = []
        saved_to: Path | None = None

        print(f"File: {path.name}")
        print(f"Type: {file_type}")

        if file_type == "image":
            objects = vision.analyze_image(path, model, draw_boxes=bool(args.draw_boxes))
            objects = _dedupe_by_label_keep_max(objects)
            frames_analyzed = 1

            if args.save_frames:
                out_dir = Path.cwd() / f"{path.stem}_frames"
                out_dir.mkdir(parents=True, exist_ok=True)
                if args.draw_boxes:
                    boxed = path.with_name(f"{path.stem}_boxed{path.suffix}")
                    if boxed.exists():
                        shutil.copy2(boxed, out_dir / boxed.name)
                else:
                    shutil.copy2(path, out_dir / path.name)

        else:
            frame_paths: list[Path] = []
            tmp_dir: Path | None = None
            try:
                frame_paths, tmp_dir = video.extract_frames(
                    path,
                    interval=int(args.interval),
                    show_progress=not args.no_progress,
                )
                frames_analyzed = len(frame_paths)

                all_results: list[dict] = []
                for i, fp in enumerate(frame_paths, start=1):
                    if not args.no_progress:
                        print(f"Analyzing frame {i}/{len(frame_paths)}: {fp.name}")
                    all_results.extend(vision.analyze_image(fp, model, draw_boxes=bool(args.draw_boxes)))
                objects = _dedupe_by_label_keep_max(all_results)

                if args.save_frames and tmp_dir is not None:
                    out_dir = Path.cwd() / f"{path.stem}_frames"
                    pattern = "*_boxed.jpg" if args.draw_boxes else "frame_*.jpg"
                    _copy_saved_frames(
                        src_dir=tmp_dir,
                        dest_dir=out_dir,
                        pattern=pattern,
                        show_progress=not args.no_progress,
                    )
            finally:
                video.cleanup_frames_dir(tmp_dir)

        print(f"Frames analyzed: {frames_analyzed}")
        print()
        print("Detected objects:")
        if objects:
            print(format_results(objects))
        else:
            print("  (none)")

        payload = {
            "file": path.name,
            "type": file_type,
            "frames_analyzed": frames_analyzed,
            "objects": objects,
        }

        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            save_json(payload, out_path)
            saved_to = out_path
            print()
            print(f"Results saved to: {saved_to}")

        return 0

    except MediaAnalyzerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

