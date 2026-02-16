import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from VideoAnimatorEngine import VideoAnimationEngine


def _parse_hsv_triplet(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("HSV triplet must be in the form H,S,V")
    return tuple(int(p) for p in parts)


def _expand_paths(items: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for item in items:
        if any(ch in item for ch in ["*", "?", "["]):
            matches = [str(p) for p in Path().glob(item)]
            paths.extend(matches)
        else:
            paths.append(item)
    return paths


def _load_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    if frame_index < 0:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _annotate_frame(frame, detection: Tuple[int, int, int]):
    x, y, width = detection
    r = max(1, int(round(width / 2)))
    out = frame.copy()
    cv2.circle(out, (x, y), r, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(out, (x, y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(out, f"x={x}, y={y}, w={width}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def main() -> int:
    video_paths = [
        str(ROOT / "tests" / "data" / "videos" / "IMG_0056.MOV"),
    ]
    frame_index = 600
    hsv_lower = "0,0,233"
    hsv_upper = "255,20,255"
    min_area = 5
    max_area = 250
    min_circularity = 0.2
    kernel_size = 5
    show = True
    out_dir_value = None

    paths = _expand_paths(video_paths)
    if not paths:
        print("No paths matched.")
        return 2

    hsv_lower = _parse_hsv_triplet(hsv_lower)
    hsv_upper = _parse_hsv_triplet(hsv_upper)

    engine = VideoAnimationEngine(capture=None)
    out_dir = Path(out_dir_value).resolve() if out_dir_value else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in paths:
        video_path = os.path.abspath(raw_path)
        if not os.path.exists(video_path):
            print(f"Missing: {video_path}")
            continue

        frame = _load_frame(video_path, frame_index)
        if frame is None:
            print(f"Failed to read frame from: {video_path}")
            continue

        detection = engine.find_golf_ball_in_keyframe(
            frame,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            min_area=min_area,
            max_area=max_area,
            min_circularity=min_circularity,
            kernel_size=kernel_size,
        )

        if detection is None:
            print(f"No ball detected: {video_path}")
            continue

        x, y, width = detection
        print(f"Detected in {video_path}: x={x}, y={y}, width={width}")

        if show or out_dir:
            annotated = _annotate_frame(frame, detection)
            if out_dir:
                out_path = out_dir / f"{Path(video_path).stem}_ball.png"
                cv2.imwrite(str(out_path), annotated)
            if show:
                cv2.imshow("Golf Ball Detection", annotated)
                cv2.waitKey(0)

    if show:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
