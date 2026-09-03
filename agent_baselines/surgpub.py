"""SurgPub-Video record normalization.

The public dataset has appeared in a few JSON variants.  This adapter accepts
the MedGRPO-style ``conversations``/``video`` format as well as the more
convenient ``question``/``answer`` format and converts both to ``VideoRequest``.
It does not copy or modify the dataset media.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .controller import VideoRequest


def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(loaded, dict):
        for key in ("data", "records", "items", "annotations"):
            if isinstance(loaded.get(key), list):
                loaded = loaded[key]
                break
        else:
            loaded = [loaded]
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON list/object or JSONL file: {path}")
    return [item for item in loaded if isinstance(item, dict)]


def _first_text(record: Mapping[str, Any], role: str) -> Optional[str]:
    conversations = record.get("conversations") or record.get("conversation")
    if not isinstance(conversations, list):
        return None
    accepted = {"human", "user"} if role == "question" else {"gpt", "assistant", "model"}
    for item in conversations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from", item.get("role", ""))).lower()
        if source in accepted:
            value = item.get("value", item.get("content", item.get("text")))
            if value is not None:
                return str(value).replace("<video>\n", "").replace("<video>", "").strip()
    return None


def _as_path(value: Any, data_root: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    path = Path(str(value).replace("file://", "", 1))
    if data_root is not None and not path.is_absolute():
        path = data_root / path
    return str(path)


def _path_list(value: Any, data_root: Optional[Path]) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [path for item in values if (path := _as_path(item, data_root)) is not None]


def _number(record: Mapping[str, Any], metadata: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = record.get(key, metadata.get(key))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _normalize_record(
    record: Mapping[str, Any],
    index: int,
    data_root: Optional[Path],
) -> VideoRequest:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    video_value = record.get("video", record.get("frames"))
    if video_value is None:
        video_value = record.get("video_path", metadata.get("video_path"))

    frame_paths: List[str] = []
    video_path: Optional[str] = None
    if isinstance(video_value, (list, tuple)):
        frame_paths = _path_list(video_value, data_root)
    elif video_value is not None:
        video_path = _as_path(video_value, data_root)

    if not frame_paths:
        frame_paths = _path_list(record.get("frame_paths", metadata.get("frame_paths")), data_root)
    if video_path is None:
        video_path = _as_path(record.get("video_path", metadata.get("video_path")), data_root)

    fps = _number(record, metadata, ("fps", "video_fps", "frame_rate"))
    start_sec = _number(record, metadata, ("start_sec", "video_start", "clip_start"))
    end_sec = _number(record, metadata, ("end_sec", "video_end", "clip_end"))
    if start_sec is None or end_sec is None:
        start_frame = _number(record, metadata, ("start_frame", "video_start_frame"))
        end_frame = _number(record, metadata, ("end_frame", "video_end_frame"))
        if fps and start_sec is None and start_frame is not None:
            start_sec = start_frame / fps
        if fps and end_sec is None and end_frame is not None:
            end_sec = end_frame / fps

    video_id = str(
        record.get("video_id")
        or metadata.get("video_id")
        or (Path(video_path).stem if video_path else (Path(frame_paths[0]).parent.name if frame_paths else f"video_{index}"))
    )
    qid = str(record.get("qid") or record.get("question_id") or record.get("id") or f"surgpub_{index:06d}")
    question = record.get("question") or record.get("query") or _first_text(record, "question")
    if question is None:
        raise ValueError(f"Record {index} has no question field")
    answer = record.get("answer") or record.get("label") or _first_text(record, "answer")

    normalized_metadata: Dict[str, Any] = {
        "dataset": "SurgPub-Video",
        "source_index": index,
        "answer": answer,
        "qa_type": record.get("qa_type"),
        "data_source": record.get("data_source"),
        "raw_metadata": dict(metadata),
    }
    if frame_paths:
        normalized_metadata["frame_paths"] = frame_paths
    if video_path:
        normalized_metadata["video_path"] = video_path
    if fps is not None:
        normalized_metadata["fps"] = fps

    return VideoRequest(
        qid=qid,
        video_id=video_id,
        question=str(question),
        video_path=video_path,
        fps=fps,
        start_sec=start_sec,
        end_sec=end_sec,
        track=record.get("track") or record.get("task") or record.get("qa_type"),
        metadata=normalized_metadata,
        frame_paths=tuple(frame_paths),
    )


def load_surgpub_requests(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    limit: Optional[int] = None,
) -> List[VideoRequest]:
    """Load SurgPub-Video JSON/JSONL into canonical controller requests."""

    root = Path(data_root) if data_root is not None else None
    records = _read_json_or_jsonl(path)
    if limit is not None:
        records = records[:limit]
    return [_normalize_record(record, index, root) for index, record in enumerate(records)]


def request_to_medgrpo_record(request: VideoRequest) -> Dict[str, Any]:
    """Convert a canonical request to the official MedGRPO JSON schema."""

    video: Any = list(request.frame_paths) if request.frame_paths else request.video_path
    if video is None:
        raise ValueError(f"Request {request.qid} has neither frame_paths nor video_path")
    metadata = dict(request.metadata.get("raw_metadata", {}))
    metadata.update({"fps": float(request.fps or metadata.get("fps", 2.0))})
    if request.start_sec is not None:
        metadata["start_sec"] = request.start_sec
        metadata["video_start"] = request.start_sec
    if request.end_sec is not None:
        metadata["end_sec"] = request.end_sec
        metadata["video_end"] = request.end_sec
    return {
        "id": request.qid,
        "video_id": request.video_id,
        "conversations": [
            {"from": "human", "value": f"<video>\n{request.question}"},
        ],
        "video": video,
        "metadata": metadata,
        "qa_type": request.track or "unknown",
        "data_source": "SurgPub-Video",
    }
