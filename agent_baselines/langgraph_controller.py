"""LangGraph-backed execution runtime for the surgical controller.

The model adapters stay local and framework-agnostic.  LangGraph supplies the
stateful execution layer: typed state, conditional routing, bounded cycles,
streaming updates, and an optional in-memory checkpointer.  Raw PIL frames are
kept in a process-local clip store; only serializable request/candidate
metadata enters graph state so checkpoints do not try to serialize media.
"""

from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, TypedDict

from .controller import (
    AnswerModel,
    ClipCandidate,
    ControllerOutput,
    InspectionResult,
    Inspector,
    ModelAnswer,
    Retriever,
    RuleGate,
    SurgicalController,
    VideoRequest,
)

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as error:  # Keep the lightweight baseline importable.
    MemorySaver = None  # type: ignore[assignment,misc]
    END = START = StateGraph = None  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR: Optional[BaseException] = error
else:
    _LANGGRAPH_IMPORT_ERROR = None


class GraphState(TypedDict, total=False):
    """Serializable state passed between LangGraph nodes."""

    run_id: str
    mode: Literal["direct", "retrieve", "inspect", "adaptive"]
    request: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    draft: Dict[str, Any]
    inspection: Dict[str, Any]
    gate: Dict[str, Any]
    final_answer: Dict[str, Any]
    trace: Dict[str, Any]


def _json_safe(value: Any) -> Any:
    """Convert adapter metadata to checkpoint/JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _request_payload(request: VideoRequest) -> Dict[str, Any]:
    return {
        "qid": request.qid,
        "video_id": request.video_id,
        "question": request.question,
        "video_path": request.video_path,
        "fps": request.fps,
        "start_sec": request.start_sec,
        "end_sec": request.end_sec,
        "track": request.track,
        "metadata": _json_safe(request.metadata),
        "frame_paths": list(request.frame_paths),
    }


def _request_from_payload(payload: Dict[str, Any]) -> VideoRequest:
    values = dict(payload)
    values["frame_paths"] = tuple(values.get("frame_paths", ()))
    return VideoRequest(**values)


def _candidate_payload(candidate: ClipCandidate) -> Dict[str, Any]:
    return {
        "clip_id": candidate.clip_id,
        "video_id": candidate.video_id,
        "start_sec": candidate.start_sec,
        "end_sec": candidate.end_sec,
        "score": candidate.score,
        "metadata": _json_safe(candidate.metadata),
    }


def _candidate_from_payload(payload: Dict[str, Any]) -> ClipCandidate:
    return ClipCandidate(
        clip_id=str(payload["clip_id"]),
        video_id=str(payload["video_id"]),
        start_sec=float(payload["start_sec"]),
        end_sec=float(payload["end_sec"]),
        score=float(payload["score"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _answer_payload(answer: ModelAnswer) -> Dict[str, Any]:
    return {
        "text": answer.text,
        "confidence": answer.confidence,
        "format_valid": answer.format_valid,
        "metadata": _json_safe(answer.metadata),
    }


def _answer_from_payload(payload: Dict[str, Any]) -> ModelAnswer:
    return ModelAnswer(
        text=str(payload.get("text", "")),
        confidence=payload.get("confidence"),
        format_valid=bool(payload.get("format_valid", True)),
        metadata=dict(payload.get("metadata", {})),
    )


class LangGraphSurgicalController:
    """Run the same baselines through a compiled LangGraph state machine.

    This class intentionally preserves the semantics of ``SurgicalController``
    rather than asking LangGraph to make an unconstrained ReAct decision.  The
    graph is therefore suitable for matched experiments and can later replace
    ``RuleGate`` with a learned policy without changing tool interfaces.
    """

    def __init__(
        self,
        answer_model: AnswerModel,
        retriever: Optional[Retriever] = None,
        inspector: Optional[Inspector] = None,
        mode: Literal["direct", "retrieve", "inspect", "adaptive"] = "direct",
        top_k: int = 5,
        gate: Optional[RuleGate] = None,
        max_attempts: int = 2,
        checkpointer: Any = "memory",
    ) -> None:
        if _LANGGRAPH_IMPORT_ERROR is not None:
            raise ImportError(
                "LangGraph runtime is optional. Install it with "
                "`pip install 'langgraph>=0.2,<2'`."
            ) from _LANGGRAPH_IMPORT_ERROR
        # Reuse constructor validation so the two runtimes cannot silently
        # diverge on required components or top-k constraints.
        SurgicalController(
            answer_model=answer_model,
            retriever=retriever,
            inspector=inspector,
            mode=mode,
            top_k=top_k,
            gate=gate,
        )
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.answer_model = answer_model
        self.retriever = retriever
        self.inspector = inspector
        self.mode = mode
        self.top_k = top_k
        self.gate = gate or RuleGate()
        self.max_attempts = max_attempts
        self._candidate_store: Dict[str, Dict[str, ClipCandidate]] = {}
        self.checkpointer = MemorySaver() if checkpointer == "memory" else checkpointer
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(GraphState)
        builder.add_node("route_start", self._route_start)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("route_after_retrieve", self._route_after_retrieve_node)
        builder.add_node("direct_answer", self._direct_answer)
        builder.add_node("retrieved_answer", self._retrieved_answer)
        builder.add_node("fallback_answer", self._fallback_answer)
        builder.add_node("inspect", self._inspect)
        builder.add_node("inspect_answer", self._inspect_answer)
        builder.add_node("draft", self._draft)
        builder.add_node("gate", self._gate)
        builder.add_node("route_after_gate", self._route_after_gate_node)
        builder.add_node("stop_after_draft", self._stop_after_draft)
        builder.add_node("final_answer", self._final_answer)

        builder.add_edge(START, "route_start")
        builder.add_conditional_edges(
            "route_start",
            lambda state: "direct_answer" if state["mode"] == "direct" else "retrieve",
            {"direct_answer": "direct_answer", "retrieve": "retrieve"},
        )
        builder.add_edge("direct_answer", END)
        builder.add_edge("retrieve", "route_after_retrieve")
        builder.add_conditional_edges(
            "route_after_retrieve",
            self._route_after_retrieve,
            {
                "fallback_answer": "fallback_answer",
                "retrieved_answer": "retrieved_answer",
                "inspect": "inspect",
                "draft": "draft",
            },
        )
        builder.add_edge("retrieved_answer", END)
        builder.add_edge("fallback_answer", END)
        # Inspection is shared by inspect-only and adaptive modes.  A
        # conditional edge keeps the graph unambiguous while preserving the
        # lightweight controller's semantics.
        builder.add_conditional_edges(
            "inspect",
            lambda state: "final_answer" if state["mode"] == "adaptive" else "inspect_answer",
            {"final_answer": "final_answer", "inspect_answer": "inspect_answer"},
        )
        builder.add_edge("inspect_answer", END)
        builder.add_edge("draft", "gate")
        builder.add_edge("gate", "route_after_gate")
        builder.add_conditional_edges(
            "route_after_gate",
            self._route_after_gate,
            {"inspect": "inspect", "stop_after_draft": "stop_after_draft"},
        )
        # An adaptive inspection returns to the final answer node rather than
        # re-entering the gate, which enforces at most one tool call.
        builder.add_edge("final_answer", END)
        builder.add_edge("stop_after_draft", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _route_start(state: GraphState) -> Dict[str, Any]:
        return {}

    def _retry(self, function: Callable[[], Any]) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_attempts):
            try:
                return function()
            except Exception as error:  # Retry only bounded tool/model calls.
                last_error = error
                if attempt + 1 >= self.max_attempts:
                    raise
        assert last_error is not None
        raise last_error

    @staticmethod
    def _trace_with_stage(state: GraphState, name: str, started: float) -> Dict[str, Any]:
        trace = dict(state.get("trace", {}))
        stages = dict(trace.get("stages", {}))
        stages[name] = perf_counter() - started
        trace["stages"] = stages
        return trace

    def _retrieve(self, state: GraphState) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("retrieve node requires a retriever")
        started = perf_counter()
        request = _request_from_payload(state["request"])
        candidates = self._retry(lambda: self.retriever.search(request, top_k=self.top_k))
        self._candidate_store[state["run_id"]] = {item.clip_id: item for item in candidates}
        trace = dict(state.get("trace", {}))
        trace["retrieved"] = [SurgicalController._candidate_trace(item) for item in candidates]
        trace = self._trace_with_stage({"trace": trace}, "retrieve", started)
        return {"candidates": [_candidate_payload(item) for item in candidates], "trace": trace}

    @staticmethod
    def _route_after_retrieve(state: GraphState) -> str:
        if not state.get("candidates"):
            return "fallback_answer"
        if state["mode"] == "retrieve":
            return "retrieved_answer"
        if state["mode"] == "inspect":
            return "inspect"
        return "draft"

    @staticmethod
    def _route_after_retrieve_node(state: GraphState) -> Dict[str, Any]:
        """Provide a state-producing node before the conditional route."""

        return {}

    def _candidate(self, state: GraphState) -> Optional[ClipCandidate]:
        candidates = state.get("candidates", [])
        if not candidates:
            return None
        payload = candidates[0]
        return self._candidate_store.get(state["run_id"], {}).get(
            str(payload["clip_id"]),
            _candidate_from_payload(payload),
        )

    def _call_answer(
        self,
        state: GraphState,
        *,
        stage: str,
        candidate: Optional[ClipCandidate] = None,
        evidence: Optional[str] = None,
        draft: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = perf_counter()
        request = _request_from_payload(state["request"])
        answer = SurgicalController._normalize_answer(
            self._retry(
                lambda: self.answer_model.answer(
                    request,
                    candidate=candidate,
                    evidence=evidence,
                    draft=draft,
                )
            )
        )
        trace = self._trace_with_stage(state, stage, started)
        trace["answer_stage"] = stage
        return {"final_answer": _answer_payload(answer), "trace": trace}

    def _direct_answer(self, state: GraphState) -> Dict[str, Any]:
        return self._call_answer(state, stage="direct")

    def _retrieved_answer(self, state: GraphState) -> Dict[str, Any]:
        return self._call_answer(state, stage="retrieved_top1", candidate=self._candidate(state))

    def _fallback_answer(self, state: GraphState) -> Dict[str, Any]:
        return self._call_answer(state, stage="fallback_no_retrieval")

    def _inspect(self, state: GraphState) -> Dict[str, Any]:
        if self.inspector is None:
            raise RuntimeError("inspect node requires an inspector")
        candidate = self._candidate(state)
        if candidate is None:
            raise RuntimeError("inspect node requires a retrieved candidate")
        started = perf_counter()
        request = _request_from_payload(state["request"])
        inspection = self._retry(lambda: self.inspector.inspect(request, candidate))
        if not isinstance(inspection, InspectionResult):
            inspection = InspectionResult(text=str(inspection))
        trace = dict(state.get("trace", {}))
        tool_calls = list(trace.get("tool_calls", []))
        tool_calls.append(
            {
                "name": "inspect_clip",
                "clip_id": candidate.clip_id,
                "output": inspection.text,
                "metadata": _json_safe(inspection.metadata),
            }
        )
        trace["tool_calls"] = tool_calls
        trace = self._trace_with_stage({"trace": trace}, "inspect", started)
        return {
            "inspection": {"text": inspection.text, "metadata": _json_safe(inspection.metadata)},
            "trace": trace,
        }

    def _inspect_answer(self, state: GraphState) -> Dict[str, Any]:
        # In inspect-only mode the inspector itself is the answerer.
        if state["mode"] == "inspect":
            inspection = state.get("inspection", {})
            answer = ModelAnswer(
                text=str(inspection.get("text", "")),
                metadata=dict(inspection.get("metadata", {})),
            )
            trace = dict(state.get("trace", {}))
            trace["answer_stage"] = "inspector"
            return {"final_answer": _answer_payload(answer), "trace": trace}
        return {}

    def _draft(self, state: GraphState) -> Dict[str, Any]:
        update = self._call_answer(state, stage="draft", candidate=self._candidate(state))
        return {"draft": update["final_answer"], "trace": update["trace"]}

    def _gate(self, state: GraphState) -> Dict[str, Any]:
        started = perf_counter()
        request = _request_from_payload(state["request"])
        draft = _answer_from_payload(state["draft"])
        candidates = [_candidate_from_payload(item) for item in state.get("candidates", [])]
        called, reasons = self.gate.should_inspect(request, draft, candidates)
        trace = self._trace_with_stage(state, "gate", started)
        trace["gate"] = {"called": called, "reasons": reasons}
        return {"gate": {"called": called, "reasons": reasons}, "trace": trace}

    @staticmethod
    def _route_after_gate(state: GraphState) -> str:
        return "inspect" if state.get("gate", {}).get("called") else "stop_after_draft"

    @staticmethod
    def _route_after_gate_node(state: GraphState) -> Dict[str, Any]:
        """Provide a state-producing node before the conditional route."""

        return {}

    def _stop_after_draft(self, state: GraphState) -> Dict[str, Any]:
        trace = dict(state.get("trace", {}))
        trace["answer_stage"] = "stop_after_draft"
        return {"final_answer": state["draft"], "trace": trace}

    def _final_answer(self, state: GraphState) -> Dict[str, Any]:
        draft = _answer_from_payload(state.get("draft", {}))
        evidence = str(state.get("inspection", {}).get("text", ""))
        return self._call_answer(
            state,
            stage="rethink_after_inspection",
            candidate=self._candidate(state),
            evidence=evidence,
            draft=draft.text,
        )

    def _initial_state(self, request: VideoRequest) -> GraphState:
        run_id = f"{request.qid}:{uuid.uuid4().hex}"
        self._candidate_store.setdefault(run_id, {})
        return {
            "run_id": run_id,
            "mode": self.mode,
            "request": _request_payload(request),
            "candidates": [],
            "trace": {
                "qid": request.qid,
                "video_id": request.video_id,
                "mode": self.mode,
                "framework": "langgraph",
                "tool_calls": [],
                "retrieved": [],
                "stages": {},
            },
        }

    def _config(self, run_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def run(self, request: VideoRequest) -> ControllerOutput:
        started = perf_counter()
        initial = self._initial_state(request)
        result = self.graph.invoke(initial, config=self._config(initial["run_id"]))
        answer = _answer_from_payload(result.get("final_answer", {}))
        trace = dict(result.get("trace", {}))
        trace["latency_sec"] = perf_counter() - started
        trace["checkpoint_thread_id"] = initial["run_id"]
        return ControllerOutput(answer=answer, trace=trace)

    def stream(self, request: VideoRequest) -> Iterator[Dict[str, Any]]:
        """Yield node-level updates for progress bars and experiment logging."""

        initial = self._initial_state(request)
        yield from self.graph.stream(
            initial,
            config=self._config(initial["run_id"]),
            stream_mode="updates",
        )

    def clear_run(self, run_id: str) -> None:
        """Release transient PIL-frame references after a run is archived."""

        self._candidate_store.pop(run_id, None)
