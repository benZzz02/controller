"""Small, dependency-light video/frame loading helpers.

The adapters deliberately keep decoding outside the controller.  A SurgPub
record can point either to an mp4 file or to an already extracted list of
frames; both are normalized to a list of RGB PIL images for model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


def _image_from_path(path: str | Path) -> Any:
    from PIL import Image

    value = str(path)
    if value.startswith("file://"):
        value = value[7:]
    with Image.open(value) as image:
        return image.convert("RGB").copy()


def _uniform_indices(length: int, num_frames: int) -> List[int]:
    if length <= 0:
        return []
    if num_frames <= 1:
        return [0]
    if length <= num_frames:
        return list(range(length))
    return [round(i * (length - 1) / (num_frames - 1)) for i in range(num_frames)]


def video_info(video_path: str | Path) -> Tuple[float, float, int]:
    """Return ``(fps, duration_sec, frame_count)`` using OpenCV lazily."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return fps, duration, frame_count


def load_frame_paths(
    frame_paths: Sequence[str | Path],
    num_frames: int,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    fps: Optional[float] = None,
) -> List[Any]:
    """Load and uniformly sample an extracted-frame sequence."""

    paths = list(frame_paths)
    if not paths:
        return []
    effective_fps = float(fps or 1.0)
    start = 0 if start_sec is None else max(0, int(start_sec * effective_fps))
    end = len(paths) if end_sec is None else min(len(paths), int(end_sec * effective_fps) + 1)
    if end <= start:
        end = min(len(paths), start + 1)
    selected = paths[start:end]
    return [_image_from_path(path) for path in (selected[i] for i in _uniform_indices(len(selected), num_frames))]


def load_video_frames(
    video_path: str | Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    num_frames: int = 16,
) -> List[Any]:
    """Decode uniformly sampled RGB PIL frames from a video window."""

    import cv2
    from PIL import Image

    fps, duration, frame_count = video_info(video_path)
    if frame_count <= 0:
        return []
    start = 0.0 if start_sec is None else max(0.0, start_sec)
    end = duration if end_sec is None else min(duration, max(start, end_sec))
    start_index = min(frame_count - 1, max(0, int(round(start * fps))))
    end_index = min(frame_count - 1, max(start_index, int(round(end * fps)) - 1))
    indices = _uniform_indices(end_index - start_index + 1, num_frames)
    absolute_indices = [start_index + index for index in indices]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: List[Any] = []
    try:
        for index in absolute_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
        capture.release()
    return frames


def load_frame_inputs(
    *,
    frame_paths: Optional[Sequence[str | Path]] = None,
    video_path: Optional[str | Path] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    fps: Optional[float] = None,
    num_frames: int = 16,
) -> List[Any]:
    """Load a clip from either extracted frames or a video file."""

    if frame_paths:
        return load_frame_paths(frame_paths, num_frames, start_sec, end_sec, fps)
    if video_path:
        return load_video_frames(video_path, start_sec, end_sec, num_frames)
    return []


def sample_ranges(
    *,
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Generate deterministic overlapping windows for long-video retrieval."""

    end = duration_sec if end_sec is None else min(duration_sec, end_sec)
    start = max(0.0, min(start_sec, end))
    if end <= start:
        return [(start, max(start + 1e-3, end))]
    if window_sec <= 0 or end - start <= window_sec:
        return [(start, end)]
    stride = max(1e-3, stride_sec)
    ranges: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end:
        right = min(end, cursor + window_sec)
        ranges.append((cursor, right))
        if right >= end:
            break
        cursor += stride

    if max_windows is not None and max_windows > 0 and len(ranges) > max_windows:
        indices = _uniform_indices(len(ranges), max_windows)
        ranges = [ranges[index] for index in indices]
    return ranges
