"""Model-agnostic controllers for frozen surgical video baselines."""

from .controller import (
    ClipCandidate,
    ControllerOutput,
    InspectionResult,
    ModelAnswer,
    RuleGate,
    SurgicalController,
    VideoRequest,
)
from .hf_video_answer import HuggingFaceVideoAnswerModel, SurgPubVideoAnswerModel
from .medgrpo_inspector import MedGRPOInspector
from .surgclip_retriever import SurgCLIPRetriever
from .surgpub import load_surgpub_requests, request_to_medgrpo_record

__all__ = [
    "ClipCandidate",
    "ControllerOutput",
    "InspectionResult",
    "ModelAnswer",
    "RuleGate",
    "SurgicalController",
    "VideoRequest",
    "HuggingFaceVideoAnswerModel",
    "SurgPubVideoAnswerModel",
    "MedGRPOInspector",
    "SurgCLIPRetriever",
    "load_surgpub_requests",
    "request_to_medgrpo_record",
]
