import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import pytest

from VideoAnimatorEngine import VideoAnimationEngine


def _split_video_list(videos_arg: Optional[str]) -> List[str]:
    if not videos_arg:
        return []
    parts = [p.strip() for p in videos_arg.split(",") if p.strip()]
    return [os.path.abspath(p) for p in parts]


def _load_expected_data(path: str) -> Dict[str, Dict]:
    ext = os.path.splitext(path)[1].lower()
    base_dir = os.path.dirname(os.path.abspath(path))
    expected: Dict[str, Dict] = {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        elif isinstance(data, dict):
            items = []
            for key, value in data.items():
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("video", key)
                    items.append(value)
        else:
            items = data
    elif ext == ".csv":
        items = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(row)
    else:
        raise ValueError("Expected data must be a .json or .csv file")

    for item in items:
        if not isinstance(item, dict):
            continue
        video = item.get("video") or item.get("path") or item.get("file")
        if not video:
            continue
        if not os.path.isabs(video):
            video = os.path.abspath(os.path.join(base_dir, video))
        expected[video] = item
        expected[os.path.basename(video)] = item

    return expected


def _parse_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_hsv_triplet(value) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 3:
            return tuple(int(p) for p in parts)
    return None


def test_golf_ball_detection_on_videos(videos_arg, request):
    """
    Expected JSON format:
      {"items": [{"video": "path.mp4", "frame_index": 0, "x": 123, "y": 456, "width": 20}]}
    CSV columns: video,frame_index,x,y,width,tol_xy,tol_w,hsv_lower,hsv_upper,min_area,max_area,min_circularity,kernel_size
    """
    expected_path = request.config.getoption("--expected")
    frame_index_default = request.config.getoption("--frame-index")
    video_paths = _split_video_list(videos_arg)

    if expected_path is None:
        default_expected = os.path.join(os.path.dirname(__file__), "data", "golf_ball_expected.json")
        expected_path = default_expected

    if not expected_path or not os.path.exists(expected_path):
        pytest.skip("Default expected data missing. Add tests/data/golf_ball_expected.json or use --expected.")

    expected = _load_expected_data(expected_path)
    if not video_paths:
        base_dir = os.path.dirname(os.path.abspath(expected_path))
        resolved_paths = []
        seen = set()
        for rec in expected.values():
            if not isinstance(rec, dict):
                continue
            raw_video = rec.get("video") or rec.get("path") or rec.get("file")
            if not raw_video:
                continue
            video_path = raw_video
            if not os.path.isabs(video_path):
                video_path = os.path.abspath(os.path.join(base_dir, video_path))
            if video_path not in seen:
                seen.add(video_path)
                resolved_paths.append(video_path)
        video_paths = resolved_paths
        if not video_paths:
            pytest.skip("No video paths listed in expected data. Add items or use --videos.")
    engine = VideoAnimationEngine(capture=None)

    for video_path in video_paths:
        if not os.path.exists(video_path):
            pytest.fail(f"Video not found: {video_path}")

        # rec is a dict that represents the json or csf file with expected values for the test.
        rec = expected.get(video_path) or expected.get(os.path.basename(video_path))
        if rec is None:
            pytest.fail(f"Missing expected data for video: {video_path}")

        frame_index = _parse_int(rec.get("frame_index"), default=frame_index_default)
        if frame_index is None:
            pytest.fail(f"Missing frame_index for video: {video_path}")

        hsv_lower = _parse_hsv_triplet(rec.get("hsv_lower"))
        hsv_upper = _parse_hsv_triplet(rec.get("hsv_upper"))
        min_area = _parse_int(rec.get("min_area"))
        max_area = _parse_int(rec.get("max_area"))
        min_circularity = _parse_float(rec.get("min_circularity"))
        kernel_size = _parse_int(rec.get("kernel_size"))
        tol_xy = _parse_int(rec.get("tol_xy"), default=5)
        tol_w = _parse_int(rec.get("tol_w"), default=5)

        expected_x = _parse_int(rec.get("x"))
        expected_y = _parse_int(rec.get("y"))
        expected_w = _parse_int(rec.get("width"))
        if expected_x is None or expected_y is None or expected_w is None:
            pytest.fail(f"Expected x/y/width missing for video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.fail(f"Unable to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            pytest.fail(f"Unable to read frame {frame_index} from video: {video_path}")

        result = engine.find_golf_ball_in_keyframe(
            frame,
            hsv_lower=hsv_lower or (0, 0, 180),
            hsv_upper=hsv_upper or (180, 60, 255),
            min_area=min_area or 20,
            max_area=max_area or 5000,
            min_circularity=min_circularity or 0.5,
            kernel_size=kernel_size or 5,
        )
        if result is None:
            pytest.fail(f"No ball detected in {video_path} frame {frame_index}")

        x, y, width = result
        assert abs(x - expected_x) <= tol_xy, (
            f"x mismatch for {video_path}: got {x}, expected {expected_x} (tol {tol_xy})"
        )
        assert abs(y - expected_y) <= tol_xy, (
            f"y mismatch for {video_path}: got {y}, expected {expected_y} (tol {tol_xy})"
        )
        assert abs(width - expected_w) <= tol_w, (
            f"width mismatch for {video_path}: got {width}, expected {expected_w} (tol {tol_w})"
        )


@pytest.fixture
def videos_arg(request):
    return request.config.getoption("--videos")
