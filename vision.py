from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_model():
    # YOLO will auto-download yolov11n.pt on first run.
    from ultralytics import YOLO

    return YOLO("yolov11n.pt")


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


def analyze_image(image_path: Path, model, *, draw_boxes: bool = False) -> list[dict]:
    image_path = Path(image_path)

    # Use Ultralytics for inference; use OpenCV to render boxes when requested.
    preds = model.predict(source=str(image_path), verbose=False)
    if not preds:
        return []

    p = preds[0]
    names = getattr(p, "names", {}) or {}

    results: list[dict] = []
    boxes = getattr(p, "boxes", None)
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        cls = boxes.cls
        conf = boxes.conf
        for i in range(int(len(cls))):
            class_id = int(cls[i].item()) if hasattr(cls[i], "item") else int(cls[i])
            c = float(conf[i].item()) if hasattr(conf[i], "item") else float(conf[i])
            label = str(names.get(class_id, str(class_id)))
            results.append({"label": label, "confidence": c})

    deduped = _dedupe_by_label_keep_max(results)

    if draw_boxes:
        _draw_and_save_boxes(image_path=image_path, pred=p, names=names)

    return deduped


def _draw_and_save_boxes(*, image_path: Path, pred, names: dict) -> None:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return

    boxes = getattr(pred, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return

    xyxy = boxes.xyxy
    cls = boxes.cls if getattr(boxes, "cls", None) is not None else None
    conf = boxes.conf if getattr(boxes, "conf", None) is not None else None
    if cls is None or conf is None:
        return

    for i in range(int(len(xyxy))):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
        class_id = int(cls[i].item()) if hasattr(cls[i], "item") else int(cls[i])
        c = float(conf[i].item()) if hasattr(conf[i], "item") else float(conf[i])
        label = str(names.get(class_id, str(class_id)))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label} {c:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    out_path = image_path.with_name(f"{image_path.stem}_boxed{image_path.suffix}")
    cv2.imwrite(str(out_path), img)


def analyze_frames(
    frame_paths: list[Path],
    model,
    *,
    draw_boxes: bool = False,
) -> list[dict]:
    all_results: list[dict] = []
    for p in frame_paths:
        all_results.extend(analyze_image(Path(p), model, draw_boxes=draw_boxes))
    return _dedupe_by_label_keep_max(all_results)
