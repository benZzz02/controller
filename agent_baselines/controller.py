"""Small, deterministic controller for no-training surgical video baselines.

The controller is intentionally independent of a model-serving framework.  A
retriever, a frozen answer model, and an optional frozen inspector are injected
through small Python protocols.  This lets the same experiment runner switch
between direct, retrieval-only, inspector-only, and adaptive-agent baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union


ControllerMode = Literal["direct", "retrieve", "inspect", "adaptive"]


@dataclass(frozen=True)
class VideoRequest:
    """Dataset-independent request passed through the pipeline."""

    qid: str
    video_id: str
    question: str
    video_path: Optional[str] = None
    fps: Optional[float] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    track: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_paths: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClipCandidate:
    """A retrieval result.  ``frames`` can be populated lazily by an adapter."""

    clip_id: str
    video_id: str
    start_sec: float
    end_sec: float
    score: float
    # A backend may keep frames as paths, PIL images, numpy arrays, or tensors.
    # Keeping this field unopinionated lets the retriever hand the exact same
    # sampled clip to a downstream video-language model without re-decoding it.
    frames: Sequence[Any] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAnswer:
    text: str
    confidence: Optional[float] = None
    format_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionResult:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerOutput:
    answer: ModelAnswer
    trace: Dict[str, Any]


class Retriever(Protocol):
    def search(self, request: VideoRequest, top_k: int) -> List[ClipCandidate]:
        ...


class AnswerModel(Protocol):
    def answer(
        self,
        request: VideoRequest,
        candidate: Optional[ClipCandidate] = None,
        evidence: Optional[str] = None,
        draft: Optional[str] = None,
    ) -> Union[ModelAnswer, str]:
        ...


class Inspector(Protocol):
    def inspect(self, request: VideoRequest, candidate: ClipCandidate) -> InspectionResult:
        ...


class RuleGate:
    """Fixed, validation-calibrated routing rule for the adaptive baseline."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        retrieval_margin_threshold: float = 0.05,
        temporal_keywords: Optional[Sequence[str]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.retrieval_margin_threshold = retrieval_margin_threshold
        self.temporal_keywords = tuple(
            temporal_keywords
            or (
                "before",
                "after",
                "when",
                "sequence",
                "first",
                "last",
                "之前",
                "之后",
                "何时",
                "顺序",
                "最先",
                "最后",
            )
        )

    def should_inspect(
        self,
        request: VideoRequest,
        draft: ModelAnswer,
        candidates: Sequence[ClipCandidate],
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if not draft.format_valid:
            reasons.append("invalid_answer_format")
        if draft.confidence is not None and draft.confidence < self.confidence_threshold:
            reasons.append("low_controller_confidence")
        if len(candidates) >= 2:
            margin = candidates[0].score - candidates[1].score
            if margin < self.retrieval_margin_threshold:
                reasons.append("small_retrieval_margin")
        question = request.question.lower()
        if any(keyword.lower() in question for keyword in self.temporal_keywords):
            reasons.append("temporal_question")

        return bool(reasons), reasons


class SurgicalController:
    """Orchestrate frozen components for the four no-training baselines.

    Modes:
      * ``direct``: answer from the original/global video input.
      * ``retrieve``: retrieve Top-K, pass Top-1 to the answer model.
      * ``inspect``: retrieve Top-1 and use the inspector as the answerer.
      * ``adaptive``: answer from Top-1, then optionally inspect once and
        ask the answer model for a final response.
    """

    def __init__(
        self,
        answer_model: AnswerModel,
        retriever: Optional[Retriever] = None,
        inspector: Optional[Inspector] = None,
        mode: ControllerMode = "direct",
        top_k: int = 5,
        gate: Optional[RuleGate] = None,
    ) -> None:
        if mode != "direct" and retriever is None:
            raise ValueError(f"mode={mode!r} requires a retriever")
        if mode in ("inspect", "adaptive") and inspector is None:
            raise ValueError(f"mode={mode!r} requires an inspector")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.answer_model = answer_model
        self.retriever = retriever
        self.inspector = inspector
        self.mode = mode
        self.top_k = top_k
        self.gate = gate or RuleGate()

    @staticmethod
    def _normalize_answer(value: Union[ModelAnswer, str]) -> ModelAnswer:
        if isinstance(value, ModelAnswer):
            return value
        return ModelAnswer(text=str(value))

    @staticmethod
    def _candidate_trace(candidate: ClipCandidate) -> Dict[str, Any]:
        return {
            "clip_id": candidate.clip_id,
            "video_id": candidate.video_id,
            "start_sec": candidate.start_sec,
            "end_sec": candidate.end_sec,
            "score": candidate.score,
        }

    def run(self, request: VideoRequest) -> ControllerOutput:
        started = perf_counter()
        trace: Dict[str, Any] = {
            "qid": request.qid,
            "video_id": request.video_id,
            "mode": self.mode,
            "tool_calls": [],
            "retrieved": [],
        }

        if self.mode == "direct":
            answer = self._normalize_answer(self.answer_model.answer(request))
            trace["answer_stage"] = "direct"
            trace["latency_sec"] = perf_counter() - started
            return ControllerOutput(answer=answer, trace=trace)

        assert self.retriever is not None
        candidates = self.retriever.search(request, top_k=self.top_k)
        trace["retrieved"] = [self._candidate_trace(item) for item in candidates]
        if not candidates:
            answer = self._normalize_answer(self.answer_model.answer(request))
            trace["answer_stage"] = "fallback_no_retrieval"
            trace["latency_sec"] = perf_counter() - started
            return ControllerOutput(answer=answer, trace=trace)

        top1 = candidates[0]

        if self.mode == "retrieve":
            answer = self._normalize_answer(self.answer_model.answer(request, candidate=top1))
            trace["answer_stage"] = "retrieved_top1"
            trace["latency_sec"] = perf_counter() - started
            return ControllerOutput(answer=answer, trace=trace)

        assert self.inspector is not None

        if self.mode == "inspect":
            inspection = self.inspector.inspect(request, top1)
            trace["tool_calls"].append(
                {
                    "name": "inspect_clip",
                    "clip_id": top1.clip_id,
                    "output": inspection.text,
                    "metadata": inspection.metadata,
                }
            )
            trace["answer_stage"] = "inspector"
            trace["latency_sec"] = perf_counter() - started
            return ControllerOutput(
                answer=ModelAnswer(text=inspection.text, metadata=inspection.metadata),
                trace=trace,
            )

        draft = self._normalize_answer(self.answer_model.answer(request, candidate=top1))
        should_call, reasons = self.gate.should_inspect(request, draft, candidates)
        trace["draft"] = draft.text
        trace["gate"] = {"called": should_call, "reasons": reasons}

        if should_call:
            inspection = self.inspector.inspect(request, top1)
            trace["tool_calls"].append(
                {
                    "name": "inspect_clip",
                    "clip_id": top1.clip_id,
                    "output": inspection.text,
                    "metadata": inspection.metadata,
                }
            )
            answer = self._normalize_answer(
                self.answer_model.answer(
                    request,
                    candidate=top1,
                    evidence=inspection.text,
                    draft=draft.text,
                )
            )
            trace["answer_stage"] = "rethink_after_inspection"
        else:
            answer = draft
            trace["answer_stage"] = "stop_after_draft"

        trace["latency_sec"] = perf_counter() - started
        return ControllerOutput(answer=answer, trace=trace)
