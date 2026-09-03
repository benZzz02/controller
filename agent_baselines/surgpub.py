"""SurgPub-Video record normalization.

The public dataset has appeared in a few JSON variants.  This adapter accepts
the MedGRPO-style ``conversations``/``video`` format as well as the more
convenient ``question``/``answer`` format and converts both to ``VideoRequest``.
It does not copy or modify the dataset media.
"""

from __future__ import annotations

import json
import csv
import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from .controller import VideoRequest


def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
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
                return (
                    str(value)
                    .replace("<video>\n", "")
                    .replace("<video>", "")
                    .replace("<image>\n", "")
                    .replace("<image>", "")
                    .strip()
                )
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


@lru_cache(maxsize=8)
def _video_url_map(csv_path: str) -> Dict[str, str]:
    """Map SurgPub's numeric video IDs to Wistia media hashes."""

    mapping: Dict[str, str] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("id", "")).strip()
            raw_url = html.unescape(str(row.get("video_url", "")).strip())
            if not video_id or not raw_url:
                continue
            path = urlparse(raw_url if "://" in raw_url else f"https:{raw_url}").path.rstrip("/")
            media_hash = path.split("/")[-1] if path else ""
            if media_hash and media_hash != "iframe":
                mapping.setdefault(video_id, media_hash)
    return mapping


def _mapped_video_path(
    video_id: str,
    *,
    data_root: Optional[Path],
    csv_path: Optional[str | Path],
    video_root: Optional[str | Path],
) -> Optional[str]:
    """Resolve a numeric dataset ID to an existing local original video."""

    root = Path(video_root) if video_root is not None else (data_root / "surgpub_videos" if data_root else Path("/data/SurgPub/surgpub_videos"))
    source_csv = Path(csv_path) if csv_path is not None else (data_root / "video_url.csv" if data_root else Path("/data/SurgPub/video_url.csv"))
    if not source_csv.exists() or not root.exists():
        return None
    media_hash = _video_url_map(str(source_csv)).get(str(video_id).strip())
    if not media_hash:
        return None
    for candidate in (root / f"{media_hash}.mp4", root / f"{media_hash}_original.mp4"):
        if candidate.is_file():
            return str(candidate)
    return None


def _normalize_record(
    record: Mapping[str, Any],
    index: int,
    data_root: Optional[Path],
    video_csv: Optional[str | Path],
    video_root: Optional[str | Path],
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
    if not frame_paths and data_root is not None and record.get("folder"):
        frame_dir = data_root / "frames" / str(record["folder"])
        if not list(frame_dir.glob("*.png")):
            frame_dir = frame_dir / "frames"
        frame_paths = [str(item) for item in sorted(frame_dir.glob("*.png"))]
    if video_path is None:
        video_path = _as_path(record.get("video_path", metadata.get("video_path")), data_root)

    fps = _number(record, metadata, ("fps", "video_fps", "frame_rate"))
    start_sec = _number(record, metadata, ("start_sec", "start", "video_start", "clip_start"))
    end_sec = _number(record, metadata, ("end_sec", "end", "video_end", "clip_end"))
    if start_sec is None or end_sec is None:
        start_frame = _number(record, metadata, ("start_frame", "video_start_frame"))
        end_frame = _number(record, metadata, ("end_frame", "video_end_frame"))
        if fps and start_sec is None and start_frame is not None:
            start_sec = start_frame / fps
        if fps and end_sec is None and end_frame is not None:
            end_sec = end_frame / fps

    if frame_paths and record.get("folder") and start_sec is not None and end_sec is not None:
        # Official SurgPub evaluation samples the annotated time interval from
        # the 1-fps extracted frames.
        first_frame = int(start_sec)
        last_frame = int(end_sec)
        selected = []
        for frame_path in frame_paths:
            match = re.search(r"frame_(\d+)\.[^.]+$", frame_path)
            if match and first_frame <= int(match.group(1)) <= last_frame:
                selected.append(frame_path)
        if selected:
            frame_paths = selected

    video_id = str(record.get("video_id") or metadata.get("video_id") or record.get("folder") or "").strip()
    if not video_id:
        if video_path:
            video_id = Path(video_path).stem
        elif frame_paths:
            frame_path = Path(frame_paths[0])
            # The native SurgPub layout is ``<numeric-video-id>/frames/*.png``.
            video_id = frame_path.parent.parent.name if frame_path.parent.name == "frames" else frame_path.parent.name
        else:
            video_id = f"video_{index}"
    explicit_video_path = video_path is not None
    if video_path is None and video_id.isdigit():
        video_path = _mapped_video_path(video_id, data_root=data_root, csv_path=video_csv, video_root=video_root)
    qid = str(record.get("qid") or record.get("question_id") or record.get("id") or f"surgpub_{index:06d}")
    question = record.get("question") or record.get("query") or record.get("question_closed") or _first_text(record, "question")
    if question is None:
        raise ValueError(f"Record {index} has no question field")
    answer = record.get("answer") or record.get("label") or _first_text(record, "answer")
    if answer is None and record.get("answer_key"):
        answer = record.get(str(record["answer_key"]).strip())

    normalized_metadata: Dict[str, Any] = {
        "dataset": "SurgPub-Video",
        "source_index": index,
        "answer": answer,
        "qa_type": record.get("qa_type"),
        "data_source": record.get("data_source"),
        "raw_metadata": dict(metadata),
    }
    if video_id.isdigit():
        normalized_metadata["video_id"] = video_id
    choices = {key: str(record[key]).strip() for key in ("A", "B", "C", "D") if record.get(key) not in (None, "")}
    if choices:
        normalized_metadata["choices"] = choices
    if video_path and not explicit_video_path:
        normalized_metadata["mapped_original_video"] = True
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
    video_csv: str | Path | None = None,
    video_root: str | Path | None = None,
) -> List[VideoRequest]:
    """Load SurgPub-Video JSON/JSONL into canonical controller requests."""

    root = Path(data_root) if data_root is not None else None
    records = _read_json_or_jsonl(path)
    if limit is not None:
        records = records[:limit]
    return [
        _normalize_record(record, index, root, video_csv, video_root)
        for index, record in enumerate(records)
    ]


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
