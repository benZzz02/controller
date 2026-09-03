"""Run the frozen SurgCLIP + SurgPub + MedGRPO controller baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .controller import ModelAnswer, SurgicalController, VideoRequest
from .hf_video_answer import SurgPubVideoAnswerModel
from .medgrpo_inspector import MedGRPOInspector, RemoteMedGRPOInspector
from .surgclip_retriever import SurgCLIPRetriever
from .surgpub import load_surgpub_requests


class _UnusedAnswerModel:
    """Placeholder for inspect-only mode, where the controller is not called."""

    def answer(self, request: VideoRequest, **_: Any) -> ModelAnswer:
        return ModelAnswer(text="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="SurgPub-Video JSON/JSONL annotation file")
    parser.add_argument("--data-root", default=None, help="Root used to resolve relative media paths")
    parser.add_argument("--output", required=True, help="Output JSONL prediction/trace file")
    parser.add_argument("--mode", choices=("direct", "retrieve", "inspect", "adaptive"), default="adaptive")
    parser.add_argument("--controller-model", default=None, help="SurgLLaVA-Video or compatible HF checkpoint")
    parser.add_argument("--controller-backend", choices=("auto", "qwen2_5_vl", "tinyllava"), default="auto")
    parser.add_argument("--inspector-model", default=None, help="MedGRPO HF checkpoint")
    parser.add_argument("--inspector-endpoint", default=None, help="Remote MedGRPO service URL")
    parser.add_argument("--surg-lavi-root", default=None)
    parser.add_argument("--surgpub-root", default=None)
    parser.add_argument("--medgrpo-root", default=None)
    parser.add_argument("--device-controller", default="cuda:0")
    parser.add_argument("--device-inspector", default="cuda:1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--cache-dir", default=None, help="Persistent on-demand SurgCLIP embedding cache")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--inspector-max-pixels", type=int, default=24 * 28 * 28)
    parser.add_argument("--inspector-min-pixels", type=int, default=8 * 28 * 28)
    parser.add_argument("--controller-4bit", action="store_true")
    parser.add_argument("--inspector-4bit", dest="inspector_4bit", action="store_true")
    parser.add_argument("--inspector-no-4bit", dest="inspector_4bit", action="store_false")
    parser.set_defaults(inspector_4bit=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only normalize records; do not load models")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _serialize_request(request: VideoRequest) -> dict[str, Any]:
    return {
        "qid": request.qid,
        "video_id": request.video_id,
        "question": request.question,
        "video_path": request.video_path,
        "frame_paths": list(request.frame_paths),
        "fps": request.fps,
        "start_sec": request.start_sec,
        "end_sec": request.end_sec,
        "track": request.track,
        "reference": request.metadata.get("answer"),
    }


def main() -> None:
    args = _build_parser().parse_args()
    requests = load_surgpub_requests(args.data, data_root=args.data_root, limit=args.limit)
    if args.dry_run:
        for request in requests:
            print(json.dumps(_serialize_request(request), ensure_ascii=False))
        return

    if args.mode in ("direct", "retrieve", "adaptive") and not args.controller_model:
        raise SystemExit(f"--controller-model is required for mode={args.mode}")
    if args.mode in ("inspect", "adaptive") and not (args.inspector_model or args.inspector_endpoint):
        raise SystemExit(f"--inspector-model or --inspector-endpoint is required for mode={args.mode}")

    retriever = None
    if args.mode != "direct":
        retriever = SurgCLIPRetriever(
            surg_lavi_root=args.surg_lavi_root,
            device=args.device_controller,
            num_frames=args.num_frames,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            max_windows=args.max_windows,
            cache_dir=args.cache_dir,
        )

    answer_model: Any = _UnusedAnswerModel()
    if args.controller_model:
        answer_model = SurgPubVideoAnswerModel(
            args.controller_model,
            backend=args.controller_backend,
            device=args.device_controller,
            quantized=args.controller_4bit,
            max_new_tokens=args.max_new_tokens,
            num_frames=args.num_frames,
            vision_processor_root=args.medgrpo_root,
            surgpub_root=args.surgpub_root,
        )

    inspector = None
    if args.inspector_endpoint:
        inspector = RemoteMedGRPOInspector(args.inspector_endpoint)
    elif args.inspector_model:
        inspector = MedGRPOInspector(
            args.inspector_model,
            medgrpo_root=args.medgrpo_root,
            device=args.device_inspector,
            quantized=args.inspector_4bit,
            max_new_tokens=args.max_new_tokens,
            max_pixels=args.inspector_max_pixels,
            min_pixels=args.inspector_min_pixels,
        )

    controller = SurgicalController(
        answer_model=answer_model,
        retriever=retriever,
        inspector=inspector,
        mode=args.mode,
        top_k=args.top_k,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, request in enumerate(requests):
            try:
                output = controller.run(request)
                result = {
                    **_serialize_request(request),
                    "prediction": output.answer.text,
                    "answer_metadata": output.answer.metadata,
                    "trace": output.trace,
                }
            except Exception as error:
                if not args.continue_on_error:
                    raise
                result = {
                    **_serialize_request(request),
                    "error": f"{type(error).__name__}: {error}",
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(requests)}] {request.qid}")


if __name__ == "__main__":
    main()
