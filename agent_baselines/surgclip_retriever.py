"""Frozen SurgCLIP retrieval adapter.

This uses the released ``SurgCLIP-B`` implementation from the local SurgLaVi
checkout.  It scores temporal windows against the question and returns the
Top-K windows, including the sampled PIL frames so a downstream inspector can
reuse them without decoding the video a second time.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .controller import ClipCandidate, Retriever, VideoRequest
from .video_io import load_frame_inputs, sample_ranges, video_info


def _default_surg_lavi_root() -> Path:
    return Path(__file__).resolve().parents[2] / "SurgLaVi"


class SurgCLIPRetriever:
    """Question-to-video-window retriever backed by released SurgCLIP weights."""

    def __init__(
        self,
        *,
        surg_lavi_root: str | Path | None = None,
        model_name: str = "SurgCLIP-B",
        device: str = "cuda:0",
        num_frames: int = 16,
        window_sec: float = 8.0,
        stride_sec: float = 4.0,
        max_windows: Optional[int] = 64,
        search_full_video: bool = True,
    ) -> None:
        self.surg_lavi_root = Path(surg_lavi_root) if surg_lavi_root else _default_surg_lavi_root()
        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.max_windows = max_windows
        self.search_full_video = search_full_video
        self._surgclip: Any = None
        self.model: Any = None
        self.preprocessor: Any = None
        self.tokenizer: Any = None
        # Keep embeddings on CPU so repeated VQA questions for one video do
        # not occupy additional VRAM.  Frames are loaded only for Top-K.
        self._embedding_cache: Dict[str, Any] = {}

    def _load_backend(self) -> None:
        if self.model is not None:
            return
        package_root = self.surg_lavi_root / "src" / "surgclip"
        if not package_root.exists():
            raise FileNotFoundError(
                f"SurgLaVi source not found at {package_root}. "
                "Pass --surg-lavi-root or clone the official SurgLaVi repository."
            )
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        import surgclip  # type: ignore

        self._surgclip = surgclip
        self.model, _, self.tokenizer = surgclip.load(
            self.model_name,
            device=self.device,
            num_frames=self.num_frames,
            temporal_modeling=True,
        )
        self.preprocessor = surgclip.VideoPreprocessor(num_frames=self.num_frames)

    @staticmethod
    def _duration_and_range(request: VideoRequest) -> Tuple[float, float, float]:
        start = float(request.start_sec or 0.0)
        if request.frame_paths:
            fps = float(request.fps or 1.0)
            duration = len(request.frame_paths) / fps
            end = float(request.end_sec if request.end_sec is not None else duration)
            return fps, duration, max(start, end)
        if not request.video_path:
            return float(request.fps or 30.0), 0.0, max(start, float(request.end_sec or start))
        fps, duration, _ = video_info(request.video_path)
        end = float(request.end_sec if request.end_sec is not None else duration)
        return fps, duration, max(start, end)

    def _ranges(self, request: VideoRequest) -> List[Tuple[float, float]]:
        if request.video_path and self.search_full_video:
            _, duration, _ = video_info(request.video_path)
            return sample_ranges(
                duration_sec=duration,
                window_sec=self.window_sec,
                stride_sec=self.stride_sec,
                max_windows=self.max_windows,
            )
        if request.frame_paths:
            fps, duration, end = self._duration_and_range(request)
            del fps
            return [(float(request.start_sec or 0.0), end if end > 0 else duration)]
        if not request.video_path:
            return []
        _, duration, end = self._duration_and_range(request)
        return sample_ranges(
            duration_sec=duration,
            window_sec=self.window_sec,
            stride_sec=self.stride_sec,
            start_sec=float(request.start_sec or 0.0),
            end_sec=end,
            max_windows=self.max_windows,
        )

    def _load_frames(self, request: VideoRequest, start_sec: float, end_sec: float) -> List[Any]:
        # SurgPub frame lists usually already represent the annotated clip; do
        # not apply absolute video timestamps as list indices a second time.
        if request.frame_paths and not (request.video_path and self.search_full_video):
            return load_frame_inputs(frame_paths=request.frame_paths, num_frames=self.num_frames)
        return load_frame_inputs(
            video_path=request.video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            num_frames=self.num_frames,
        )

    def _text_embedding(self, question: str) -> Any:
        import torch
        import torch.nn.functional as F

        tokens = self._surgclip.tokenize(question, self.tokenizer, device=self.device)
        with torch.inference_mode():
            _, pooled = self.model.encode_text(tokens)
            projected = self.model.text_proj(pooled)
            return F.normalize(projected, dim=-1)

    def _video_embedding(self, frames: Sequence[Any]) -> Any:
        import torch
        import torch.nn.functional as F

        video = self.preprocessor(list(frames)).to(self.device)
        with torch.inference_mode():
            _, pooled = self.model.encode_vision(video)
            projected = self.model.vision_proj(pooled)
            # Some SurgCLIP checkpoints return one projected embedding per
            # temporal token (B, T, D), while others return (B, D).
            # Retrieval needs one vector for the complete window.
            if projected.ndim == 3:
                projected = projected.mean(dim=1)
            return F.normalize(projected, dim=-1)

    def search(self, request: VideoRequest, top_k: int = 5) -> List[ClipCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._load_backend()
        text_embedding = self._text_embedding(request.question)
        candidates: List[ClipCandidate] = []
        for index, (start_sec, end_sec) in enumerate(self._ranges(request)):
            clip_id = f"{request.video_id}:{start_sec:.3f}-{end_sec:.3f}"
            was_cached = clip_id in self._embedding_cache
            frames: List[Any] = []
            if was_cached:
                video_embedding = self._embedding_cache[clip_id].to(self.device)
            else:
                frames = self._load_frames(request, start_sec, end_sec)
                if not frames:
                    continue
                video_embedding = self._video_embedding(frames)
                self._embedding_cache[clip_id] = video_embedding.detach().cpu()
            score_tensor = video_embedding @ text_embedding.T
            score = float(score_tensor.reshape(-1).mean().item())
            metadata = {
                "retriever": self.model_name,
                "video_path": request.video_path,
                "frame_paths": list(request.frame_paths),
                "fps": request.fps,
                "window_index": index,
            }
            candidates.append(
                ClipCandidate(
                    clip_id=clip_id,
                    video_id=request.video_id,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    score=score,
                    frames=tuple(frames) if not was_cached else tuple(),
                    metadata=metadata,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[:top_k]
        # The just-computed candidate may have frames in the local variable,
        # but cached candidates intentionally do not.  Materialize only the
        # returned clips for the downstream VLM.
        return [
            item
            if item.frames
            else replace(
                item,
                frames=tuple(self._load_frames(request, item.start_sec, item.end_sec)),
            )
            for item in selected
        ]
